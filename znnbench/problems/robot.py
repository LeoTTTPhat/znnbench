"""Velocity-level path tracking of redundant serial manipulators.

This is the standard robotics application of zeroing / gradient neural
dynamics (Zhang, Jin, Li, Guo, Xiao and co-workers): the end-effector
position r = f(theta) of an n-joint arm must follow a desired Cartesian path
r_d(t) in R^3 (position-only task, nominally n - 3 degrees of redundancy).
NOTE (verified, see the spec): on both benchmark arms the end-effector point
lies on the axis of the LAST joint (the tool/flange offset is along z_n), so
the last column of the position Jacobian is identically zero and the
effective redundancy is n - 4 (PUMA 560: 2, Panda: 3).  This also holds for
the PUMA table of Jin et al. (TIE 2017), whose d6 = 0.2 m lies along z5.
With the error function

    E(theta, t) = f(theta) - r_d(t)                         (B, 3)

the ZNN design formula dE/dt = -gamma Phi(E) becomes the familiar
velocity-level resolved-rate law with position feedback; for the linear
activation it is exactly eq. (12)/(13) of Jin, Li, La, Luo, IEEE TIE 64(6)
2017 ("J w = -k0 (f(theta) - r_d) + r_d_dot", k0 = gamma):

    J(theta) theta_dot = r_d_dot(t) - gamma Phi(f(theta) - r_d(t)),

solved for theta_dot with the minimum-norm pseudoinverse J^+ = J^T (J J^T)^-1.
Here J = df/dtheta in R^{3 x n} is the geometric (position) Jacobian and
dE/dt at fixed theta is -r_d_dot(t).

Kinematic models (see docs/spec/problems_robot.md for the verification log)
---------------------------------------------------------------------------
* ``robot_puma560``  -- Unimation PUMA 560, 6 revolute joints, *standard* DH
  (Corke, Robotics Toolbox for MATLAB ``mdl_puma560.m``, link parameters,
  joint limits and named poses; identical link parameters in the Python
  toolbox ``models/DH/Puma560.py`` except for the 26.45-inch pedestal d1,
  available as ``base_z=PUMA560_PEDESTAL``).  The position-only task needs a
  tool offset along z6, otherwise the wrist centre is the end-effector and
  the last three Jacobian columns vanish; the default ``tool_d = 0.2`` m is the d6 value of
  Table I of Jin et al. (TIE 2017).  ``dh="jin2017"`` reproduces that table
  verbatim (a3 = 0.4318, d4 = 0, d1 = 0.67, d6 = 0.2).
* ``robot_panda``    -- Franka Emika Panda (FER), 7 revolute joints,
  *modified* (Craig) DH from the official Franka Control Interface docs
  ("Control Interface Specification and Robot Limits -> Denavit-Hartenberg
  Parameters"), end-effector at the flange (d = 0.107 m) or, with
  ``ee="tcp"``, at the Franka-hand TCP (+0.1034 m, franka_description).

Desired paths (all anchored at the initial end-effector position
p0 = f(theta0) of the nominal configuration unless ``center`` is given,
exact time derivatives)
---------------------------------------------------------------------
* ``circle``     plane rotated 30 deg about X (Jin et al. 2017); radius 0.1 m
                 (ASSUMPTION: Jin et al. use 0.2 m about the absolute centre
                 [0.25, 0, 1.3] m; a 0.2 m circle anchored at the nominal start
                 reaches 0.4 m towards the base and drives PUMA joint 3 / Panda
                 joint 2 beyond their limits under pseudoinverse tracking).
                 The paper's setting is available as
                 ``dh='jin2017', center=(0.25, 0, 1.3), path_T=2*pi/0.2,
                 path_kw=dict(radius=0.2, phase='linear')`` with gamma = k0 = 5.
* ``lissajous``  x = A sin(2 pi a t/T + delta), y = B sin(2 pi b t/T),
                 A = B = 0.06 m, a = 3, b = 2, delta = pi/2
                 (Yang et al., Neural Networks 2025, eq. (30), Franka Panda).
* ``plum``       five-petal plum (rose-type) curve, x = r(cos phi + cos(n phi)/n),
                 y = r(sin phi + sin(n phi)/n), r = 0.1 m, n = 5
                 (Yang et al. 2025, eq. (31)).
* ``heart``      x = a(2 sin th - sin 2th), y = a(2 cos th - cos 2th) - a,
                 th = 2 pi t/T (Yang et al. 2025, eq. (29); a is not given
                 in the paper -- ASSUMPTION a = 0.03 m).
The smooth phase phi(t) = 2 pi sin^2(pi t / (2 T)) (Yang et al. 2025, eq. (31);
also the standard choice in Zhang's repetitive-motion papers) closes the path
at t = T with zero initial and final path speed.

x_star is None (redundant problem); the residual ||E|| is the Cartesian
position error in metres.  ``joint_drift`` / ``StateRecorder`` provide the
repetitive-motion (joint-drift) metric ||theta(T) - theta(0)||.
"""
from __future__ import annotations

import numpy as np

from ..core import Discretizer, Problem

PI = np.pi

# ---------------------------------------------------------------------------
# kinematic tables
# ---------------------------------------------------------------------------
# Corke, Robotics Toolbox for MATLAB, models/mdl_puma560.m (standard DH):
#   L(1) = Revolute('d', 0,       'a', 0,      'alpha',  pi/2)
#   L(2) = Revolute('d', 0,       'a', 0.4318, 'alpha',  0)
#   L(3) = Revolute('d', 0.15005, 'a', 0.0203, 'alpha', -pi/2)
#   L(4) = Revolute('d', 0.4318,  'a', 0,      'alpha',  pi/2)
#   L(5) = Revolute('d', 0,       'a', 0,      'alpha', -pi/2)
#   L(6) = Revolute('d', 0,       'a', 0,      'alpha',  0)
#   qz = [0 0 0 0 0 0]; qr = [0 pi/2 -pi/2 0 0 0]; qs = [0 0 -pi/2 0 0 0];
#   qn = [0 pi/4 pi 0 pi/4 0]
# Joint limits: the 'qlim' entries of mdl_puma560.m (degrees)
#   [-160,160], [-45,225], [-225,45], [-110,170], [-100,100], [-266,266].
# (The Python toolbox Puma560.py keeps these as comments but uses the active
#  limits [-160,160], [-110,110], [-135,135], [-266,266], [-100,100], [-266,266];
#  it cites Wang & Chen, IEEE T-RA 7(4) 1991 for joint limits.  Under the Python
#  limits qn's theta3 = +-pi is out of range, so the MATLAB limits are used.)
PUMA560_CORKE = dict(
    convention="standard",
    a=[0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0],
    d=[0.0, 0.0, 0.15005, 0.4318, 0.0, 0.0],
    alpha=[PI / 2, 0.0, -PI / 2, PI / 2, -PI / 2, 0.0],
    qlim=np.deg2rad([[-160, 160], [-45, 225], [-225, 45], [-110, 170], [-100, 100],
                     [-266, 266]]),
    source="Corke, Robotics Toolbox for MATLAB, mdl_puma560.m (standard DH, qlim)",
)
PUMA560_PEDESTAL = 26.45 * 0.0254   # Python toolbox Puma560.py: base = 26.45 * inch

# Jin, Li, La, Luo, IEEE TIE 64(6):4710-4720, 2017, Table I "PUMA 560 D-H
# parameters" (standard DH, d6 = 0.2 m is the tool length):
#   a = [0, 0.4318, 0.4318, 0, 0, 0], alpha = [pi/2, 0, -pi/2, pi/2, -pi/2, 0],
#   d = [0.67, 0, 0.15005, 0, 0, 0.2]
PUMA560_JIN2017 = dict(
    convention="standard",
    a=[0.0, 0.4318, 0.4318, 0.0, 0.0, 0.0],
    d=[0.67, 0.0, 0.15005, 0.0, 0.0, 0.2],
    alpha=[PI / 2, 0.0, -PI / 2, PI / 2, -PI / 2, 0.0],
    qlim=PUMA560_CORKE["qlim"],
    source="Jin, Li, La, Luo, IEEE TIE 2017, Table I",
)

# Franka Control Interface documentation, "Denavit-Hartenberg Parameters"
# (Craig's convention; row i lists a_{i-1}, d_i, alpha_{i-1}, theta_i):
#   Joint 1: a 0       d 0.333  alpha  0
#   Joint 2: a 0       d 0      alpha -pi/2
#   Joint 3: a 0       d 0.316  alpha  pi/2
#   Joint 4: a 0.0825  d 0      alpha  pi/2
#   Joint 5: a -0.0825 d 0.384  alpha -pi/2
#   Joint 6: a 0       d 0      alpha  pi/2
#   Joint 7: a 0.088   d 0      alpha  pi/2
#   Flange : a 0       d 0.107  alpha  0      (theta = 0)
# FER joint limits (same page): q_max = [2.8973 1.7628 2.8973 -0.0698 2.8973
# 3.7525 2.8973], q_min = [-2.8973 -1.7628 -2.8973 -3.0718 -2.8973 -0.0175 -2.8973]
# (The page labels the DH table "Franka Research 3"; the same kinematics are in
#  franka_description/robots/common/franka_arm.xacro, joint1..joint8 origins,
#  and in Corke's Python Panda.py -- cross-checked numerically in the tests.)
PANDA_FRANKA = dict(
    convention="modified",
    a=[0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088],
    d=[0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0],
    alpha=[0.0, -PI / 2, PI / 2, PI / 2, -PI / 2, PI / 2, PI / 2],
    flange_d=0.107,
    qlim=np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                   [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
                   [-2.8973, 2.8973]]),
    source="Franka Control Interface docs, robot_specifications (Craig DH)",
)
PANDA_TCP_D = 0.1034    # franka_description franka_hand.xacro: tcp_xyz = '0 0 0.1034'

# nominal initial configurations
# Corke qn: both mdl_puma560.m and the Python Puma560.py list
# [0 pi/4 pi 0 pi/4 0].  ASSUMPTION (wrapping only): we use theta3 = -pi, the
# same pose, because +pi violates the mdl_puma560.m joint-3 limit [-225, 45] deg.
PUMA560_QN = np.array([0.0, PI / 4, -PI, 0.0, PI / 4, 0.0])
# ASSUMPTION (jin2017 table only): Jin et al. start from random configurations.
# In their table the forearm lies along x3 instead of z3 (Corke), i.e. it is
# rotated by +pi/2 about z2, so Corke's qn posture corresponds to theta3 + pi/2.
PUMA560_QN_JIN2017 = PUMA560_QN + np.array([0.0, 0.0, PI / 2, 0.0, 0.0, 0.0])
PANDA_READY = np.array([0.0, -PI / 4, 0.0, -3 * PI / 4, 0.0, PI / 2, PI / 4])  # libfranka


# ---------------------------------------------------------------------------
# batched forward kinematics and geometric (position) Jacobian
# ---------------------------------------------------------------------------
class SerialArm:
    """All-revolute serial arm described by standard or modified DH tables.

    Standard DH (Corke ``RevoluteDH``):   A_i = Rz(th_i) Tz(d_i) Tx(a_i) Rx(alpha_i);
        joint i rotates about z_{i-1}.
    Modified DH (Craig, ``RevoluteMDH``): A_i = Rx(alpha_{i-1}) Tx(a_{i-1}) Rz(th_i) Tz(d_i);
        joint i rotates about z_i.
    The end-effector point is the origin of the last frame translated by
    ``tool_d`` along its z axis, plus a constant base height ``base_z``.
    Position Jacobian column i: z_axis_i x (p_e - o_axis_i).
    """

    def __init__(self, name, convention, a, d, alpha, offset=None, tool_d=0.0, base_z=0.0,
                 qlim=None):
        if convention not in ("standard", "modified"):
            raise ValueError(convention)
        self.name = name
        self.convention = convention
        self.a = np.asarray(a, float)
        self.d = np.asarray(d, float)
        self.alpha = np.asarray(alpha, float)
        self.n = len(self.a)
        self.offset = np.zeros(self.n) if offset is None else np.asarray(offset, float)
        self.tool_d = float(tool_d)
        self.base_z = float(base_z)
        self.qlim = None if qlim is None else np.asarray(qlim, float)
        self._ca, self._sa = np.cos(self.alpha), np.sin(self.alpha)

    def _link(self, i, th):
        """Rotation (B,3,3) and translation (B,3) of link transform i."""
        c, s = np.cos(th), np.sin(th)
        ca, sa, a, d = self._ca[i], self._sa[i], self.a[i], self.d[i]
        B = th.shape[0]
        R = np.empty((B, 3, 3))
        if self.convention == "standard":
            R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = c, -s * ca, s * sa
            R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = s, c * ca, -c * sa
            R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = 0.0, sa, ca
            p = np.stack([a * c, a * s, np.full(B, d)], axis=1)
        else:
            R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = c, -s, 0.0
            R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = s * ca, c * ca, -sa
            R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = s * sa, c * sa, ca
            p = np.stack([np.full(B, a), np.full(B, -sa * d), np.full(B, ca * d)], axis=1)
        return R, p

    def fk_jac(self, theta):
        """End-effector positions (B,3) and position Jacobians (B,3,n)."""
        theta = np.atleast_2d(np.asarray(theta, float))
        B = theta.shape[0]
        q = theta + self.offset
        R = np.broadcast_to(np.eye(3), (B, 3, 3))
        p = np.zeros((B, 3))
        p[:, 2] = self.base_z
        axes = np.empty((B, self.n, 3))
        origins = np.empty((B, self.n, 3))
        for i in range(self.n):
            if self.convention == "standard":         # joint i turns about z_{i-1}
                axes[:, i], origins[:, i] = R[:, :, 2], p
            Ri, pi_ = self._link(i, q[:, i])
            p = p + np.einsum("bij,bj->bi", R, pi_)
            R = R @ Ri
            if self.convention == "modified":         # joint i turns about z_i
                axes[:, i], origins[:, i] = R[:, :, 2], p
        pe = p + self.tool_d * R[:, :, 2]
        J = np.cross(axes, pe[:, None, :] - origins).transpose(0, 2, 1)
        return pe, J

    def fk(self, theta):
        return self.fk_jac(theta)[0]

    def jacobian(self, theta):
        return self.fk_jac(theta)[1]


def make_arm(robot="puma560", dh=None, tool_d=None, ee="flange", base_z=None):
    """Factory for the two benchmark arms (see module docstring).

    robot='puma560': dh='corke' (default) or 'jin2017'; tool_d = tool length
    along z6 (default 0.2 m).  robot='panda': dh='franka' (default, the only
    table); ee='flange' | 'tcp', or an explicit tool_d beyond the flange."""
    if robot == "puma560":
        if dh in (None, "corke"):
            tab = PUMA560_CORKE
            tool = 0.2 if tool_d is None else tool_d       # ASSUMPTION: Jin et al. d6
            bz = 0.0 if base_z is None else base_z          # MATLAB toolbox (no pedestal)
            return SerialArm("puma560", tab["convention"], tab["a"], tab["d"], tab["alpha"],
                             tool_d=tool, base_z=bz, qlim=tab["qlim"])
        if dh == "jin2017":
            tab = PUMA560_JIN2017
            d = list(tab["d"])
            if tool_d is not None:
                d[5] = tool_d
            return SerialArm("puma560_jin2017", tab["convention"], tab["a"], d, tab["alpha"],
                             tool_d=0.0, base_z=0.0 if base_z is None else base_z,
                             qlim=tab["qlim"])
        raise ValueError(f"unknown PUMA560 table {dh!r}")
    if robot == "panda":
        if dh not in (None, "franka"):
            raise ValueError(f"unknown Panda table {dh!r}")
        tab = PANDA_FRANKA
        if tool_d is None:
            tool_d = {"flange": 0.0, "tcp": PANDA_TCP_D}[ee]
        return SerialArm("panda", tab["convention"], tab["a"], tab["d"], tab["alpha"],
                         tool_d=tab["flange_d"] + tool_d,
                         base_z=0.0 if base_z is None else base_z, qlim=tab["qlim"])
    raise ValueError(f"unknown robot {robot!r}")


# ---------------------------------------------------------------------------
# desired paths (offset from the start point, with exact derivatives)
# ---------------------------------------------------------------------------
def _phase(t, T, kind):
    """phi(t), phi_dot(t).  'smooth': 2 pi sin^2(pi t/(2T)); 'linear': 2 pi t / T."""
    if kind == "smooth":
        return 2 * PI * np.sin(PI * t / (2 * T)) ** 2, (PI ** 2 / T) * np.sin(PI * t / T)
    if kind == "linear":
        return 2 * PI * t / T, 2 * PI / T
    raise ValueError(kind)


class Path:
    """Planar curve (u(t), v(t)) in the plane spanned by e_u = X and
    e_v = Rx(tilt) Y, shifted so that it passes through the start point at t=0."""

    key = "path"

    def __init__(self, T=10.0, tilt=0.0, phase="smooth"):
        self.T, self.tilt, self.phase = float(T), float(tilt), phase
        self.eu = np.array([1.0, 0.0, 0.0])
        self.ev = np.array([0.0, np.cos(self.tilt), np.sin(self.tilt)])
        self._uv0 = self._uv(0.0)[0]

    def _uv(self, t):
        """Return ((u, v), (u_dot, v_dot)) before shifting."""
        raise NotImplementedError

    def __call__(self, t):
        (u, v), _ = self._uv(t)
        return (u - self._uv0[0]) * self.eu + (v - self._uv0[1]) * self.ev

    def dot(self, t):
        _, (ud, vd) = self._uv(t)
        return ud * self.eu + vd * self.ev

    def params(self):
        return {"T": self.T, "tilt": self.tilt, "phase": self.phase}

    def describe(self):
        return {"key": self.key, **self.params()}


class Circle(Path):
    """Circle of radius R: (u, v) = R (cos phi, sin phi), i.e. the classic
    r_d = c + R [cos phi, sin phi cos(tilt), sin phi sin(tilt)].  Jin, Li, La,
    Luo, IEEE TIE 2017, Sec. VI-B: centre c = [0.25, 0, 1.3] m, radius 0.2 m,
    plane revolved 30 deg about X, constant angular speed 0.2 rad/s
    (= phase='linear' with T = 2 pi / 0.2), k0 = 5, random initial
    configuration off the path.  ASSUMPTIONS: the sign of the 30-deg rotation
    (+30 deg here) and the start phase phi(0) = 0 are not stated in the paper;
    default radius 0.1 m (see module docstring)."""

    key = "circle"

    def __init__(self, radius=0.1, tilt=PI / 6, T=10.0, phase="smooth"):
        self.R = float(radius)
        super().__init__(T=T, tilt=tilt, phase=phase)

    def _uv(self, t):
        ph, phd = _phase(t, self.T, self.phase)
        c, s = np.cos(ph), np.sin(ph)
        return (self.R * c, self.R * s), (-self.R * s * phd, self.R * c * phd)

    def params(self):
        return {**super().params(), "radius": self.R}


class Lissajous(Path):
    """Yang et al., Neural Networks 2025, eq. (30):
    x = A sin(2 pi a t/T + delta), y = B sin(2 pi b t/T), z = z0."""

    key = "lissajous"

    def __init__(self, A=0.06, B=0.06, a=3, b=2, delta=PI / 2, T=10.0, tilt=0.0):
        self.A, self.B, self.a, self.b, self.delta = float(A), float(B), a, b, float(delta)
        super().__init__(T=T, tilt=tilt, phase="linear")

    def _uv(self, t):
        w = 2 * PI / self.T
        aa, bb = w * self.a * t + self.delta, w * self.b * t
        return ((self.A * np.sin(aa), self.B * np.sin(bb)),
                (self.A * w * self.a * np.cos(aa), self.B * w * self.b * np.cos(bb)))

    def params(self):
        return {**super().params(), "A": self.A, "B": self.B, "a": self.a, "b": self.b,
                "delta": self.delta}


class Plum(Path):
    """Five-petal plum (rose-type) curve, Yang et al. 2025, eq. (31):
    x = r (cos phi + cos(n phi)/n), y = r (sin phi + sin(n phi)/n),
    phi = 2 pi sin^2(pi t / (2 T)), r = 0.1, n = 5."""

    key = "plum"

    def __init__(self, r=0.1, n=5, T=10.0, tilt=0.0, phase="smooth"):
        self.r, self.nl = float(r), int(n)
        super().__init__(T=T, tilt=tilt, phase=phase)

    def _uv(self, t):
        ph, phd = _phase(t, self.T, self.phase)
        r, n = self.r, self.nl
        u = r * (np.cos(ph) + np.cos(n * ph) / n)
        v = r * (np.sin(ph) + np.sin(n * ph) / n)
        ud = -r * (np.sin(ph) + np.sin(n * ph)) * phd
        vd = r * (np.cos(ph) + np.cos(n * ph)) * phd
        return (u, v), (ud, vd)

    def params(self):
        return {**super().params(), "r": self.r, "n": self.nl}


class Heart(Path):
    """Heart-shaped curve, Yang et al. 2025, eq. (29):
    x = a (2 sin th - sin 2 th), y = a (2 cos th - cos 2 th) - a, th = 2 pi t/T.
    ASSUMPTION: the paper leaves the scale a unspecified; a = 0.03 m here."""

    key = "heart"

    def __init__(self, a=0.03, T=10.0, tilt=0.0, phase="linear"):
        self.a = float(a)
        super().__init__(T=T, tilt=tilt, phase=phase)

    def _uv(self, t):
        th, thd = _phase(t, self.T, self.phase)
        a = self.a
        u = a * (2 * np.sin(th) - np.sin(2 * th))
        v = a * (2 * np.cos(th) - np.cos(2 * th)) - a
        ud = a * (2 * np.cos(th) - 2 * np.cos(2 * th)) * thd
        vd = a * (-2 * np.sin(th) + 2 * np.sin(2 * th)) * thd
        return (u, v), (ud, vd)

    def params(self):
        return {**super().params(), "a": self.a}


PATHS = {"circle": Circle, "lissajous": Lissajous, "plum": Plum, "heart": Heart}


def make_path(key, T=10.0, **kw):
    return PATHS[key](T=T, **kw)


# ---------------------------------------------------------------------------
# the problem
# ---------------------------------------------------------------------------
class ManipulatorTracking(Problem):
    """Position-level path tracking of a redundant arm, solved at velocity level.

    State x = theta (B, n_joints); error E = f(theta) - r_d(t) (B, 3);
    J = position Jacobian (B, 3, n); solve_J returns the minimum-norm solution
    J^T (J J^T + lambda^2 I)^{-1} r (lambda = ``damping``, default 0 = exact
    pseudoinverse); dE/dt = -r_d_dot(t).

    Parameters
    ----------
    robot : 'puma560' | 'panda'
    path  : 'circle' | 'lissajous' | 'plum' | 'heart'
    T     : horizon (s); the path period defaults to T.
    theta0 : nominal initial configuration (default: Corke's qn for the
             PUMA 560, libfranka's ready pose for the Panda); the path starts
             at f(theta0).
    x0_spread : half-width (rad) of the uniform perturbation of theta0 used by
             ``x0`` (0 = every trajectory starts exactly on the path).
    damping : Tikhonov parameter lambda of the damped pseudoinverse.
    path_kw : extra keyword arguments of the path class (radius, tilt, ...).
    center : None (default) anchors the path at p0 = f(theta0), so every
             trajectory starts on the path; a 3-vector instead places the
             curve's origin (u, v) = (0, 0) at this absolute point, as in the
             source papers (e.g. Jin et al. 2017: circle centre [0.25, 0, 1.3]
             m), and theta0 then generally starts OFF the path.
    dh, tool_d, ee, base_z : kinematic options, see ``make_arm``.
    """

    family = "robot"

    def __init__(self, robot="puma560", path="circle", T=10.0, path_T=None, theta0=None,
                 x0_spread=0.0, damping=0.0, dh=None, tool_d=None, ee="flange",
                 base_z=None, path_kw=None, center=None):
        self.robot = robot
        self.arm = make_arm(robot, dh=dh, tool_d=tool_d, ee=ee, base_z=base_z)
        if theta0 is None:
            if robot == "puma560":
                theta0 = PUMA560_QN_JIN2017 if dh == "jin2017" else PUMA560_QN
            else:
                theta0 = PANDA_READY
        self.theta0 = np.asarray(theta0, float).copy()
        if self.theta0.shape != (self.arm.n,):
            raise ValueError("theta0 has wrong length")
        self.T = float(T)
        self.path = make_path(path, T=self.T if path_T is None else float(path_T),
                              **(path_kw or {}))
        self.p0 = self.arm.fk(self.theta0[None])[0]
        if center is None:                      # anchored: r_d(0) = p0
            self.center = None
            self.origin = self.p0.copy()
        else:                                   # absolute: r_d = c + u e_u + v e_v
            self.center = np.asarray(center, float).reshape(3).copy()
            u0, v0 = self.path._uv0
            self.origin = self.center + u0 * self.path.eu + v0 * self.path.ev
        self.n_x = self.arm.n
        self.m = 3
        self.x0_spread = float(x0_spread)
        self.damping = float(damping)
        if dh is None:
            dh = "corke" if robot == "puma560" else "franka"
        self.name = (f"robot_{robot}_{path}" + ("_jin2017" if dh == "jin2017" else "")
                     + ("" if self.center is None else "_abs"))
        self.cfg = dict(robot=robot, path=path, dh=dh, T=self.T, damping=self.damping,
                        x0_spread=self.x0_spread, tool_d=self.arm.tool_d, ee=ee,
                        path_params=self.path.params(), theta0=self.theta0.tolist(),
                        center=None if self.center is None else self.center.tolist())
        self._ck = None
        self._cv = None

    # --- kinematics with a one-entry cache (E, solve_J, JT share x) ---------
    def _kin(self, x):
        if self._ck is not None and self._ck.shape == x.shape and np.array_equal(self._ck, x):
            return self._cv
        pe, J = self.arm.fk_jac(x)
        self._ck, self._cv = np.array(x, dtype=float, copy=True), (pe, J)
        return pe, J

    def fk(self, theta):
        """End-effector position f(theta), shape (B, 3)."""
        return self._kin(np.atleast_2d(theta))[0].copy()

    def jacobian(self, theta):
        """Position Jacobian df/dtheta, shape (B, 3, n)."""
        return self._kin(np.atleast_2d(theta))[1].copy()

    def manipulability(self, theta):
        """Yoshikawa measure sqrt(det(J J^T)) per trajectory, shape (B,)."""
        J = self.jacobian(theta)
        return np.sqrt(np.maximum(np.linalg.det(J @ J.transpose(0, 2, 1)), 0.0))

    # --- desired path ---------------------------------------------------------
    def rd(self, t):
        return self.origin + self.path(t)

    def rd_dot(self, t):
        return self.path.dot(t)

    # --- Problem API ----------------------------------------------------------
    def data(self, t):
        return {"t": np.float64(t), "rd": self.rd(t)}

    def E(self, x, d):
        return self._kin(x)[0] - d["rd"]

    def dEdt(self, x, t):
        return np.broadcast_to(-self.rd_dot(t), (x.shape[0], 3)).copy()

    def solve_J(self, x, d, r):
        J = self._kin(x)[1]
        A = J @ J.transpose(0, 2, 1)
        if self.damping:
            A = A + self.damping ** 2 * np.eye(3)
        try:
            y = np.linalg.solve(A, r[..., None])[..., 0]
        except np.linalg.LinAlgError:                     # exactly singular J J^T
            return np.einsum("bnm,bm->bn", np.linalg.pinv(J, rcond=1e-12), r)
        return np.einsum("bmn,bm->bn", J, y)

    def JT(self, x, d, v):
        return np.einsum("bmn,bm->bn", self._kin(x)[1], v)

    def x_star(self, t):
        return None

    def x0(self, rng, B):
        th = np.broadcast_to(self.theta0, (B, self.n_x)).copy()
        if self.x0_spread:
            th += rng.uniform(-self.x0_spread, self.x0_spread, size=th.shape)
        return th

    def capabilities(self):
        return {"robot_kinematics", "redundant"}

    def describe(self):
        return {**super().describe(), **self.cfg}


# ---------------------------------------------------------------------------
# repetitive-motion (joint-drift) metric
# ---------------------------------------------------------------------------
def joint_drift(theta_final, theta_initial, per_joint=False):
    """Joint-angle drift theta(T) - theta(0) of repetitive motion planning
    (Zhang et al.): returns the per-joint drift (B, n) if ``per_joint`` else its
    Euclidean norm per trajectory (B,).  Only meaningful for closed paths
    (circle / plum with the smooth phase, or Lissajous / heart at t = T)."""
    dth = np.atleast_2d(theta_final) - np.atleast_2d(theta_initial)
    return dth if per_joint else np.linalg.norm(dth, axis=-1)


def qlim_violation(theta, qlim):
    """Largest joint-limit violation (rad, >= 0) per trajectory, shape (B,).
    The problem itself does not enforce limits (pure pseudoinverse tracking);
    this helper reports them, e.g. on states stored by ``StateRecorder``."""
    th = np.atleast_2d(theta)
    q = np.asarray(qlim, float)
    v = np.maximum(q[:, 0] - th, 0.0) + np.maximum(th - q[:, 1], 0.0)
    return v.max(axis=-1)


class StateRecorder(Discretizer):
    """Transparent wrapper around a discretizer that stores the joint states
    (``simulate`` itself only returns residual traces).  ``x_first`` is the
    initial state, ``x_last`` the state after the final step, and ``states``
    holds every ``every``-th state (if ``every`` > 0)."""

    def __init__(self, inner, every=0):
        self.inner = inner
        self.every = int(every)
        self.name = getattr(inner, "name", "disc")
        self.causal = getattr(inner, "causal", True)
        self.expected_order = getattr(inner, "expected_order", None)
        self.implicit = getattr(inner, "implicit", False)
        self.reset()

    def bind_env(self, **kw):
        if callable(getattr(self.inner, "bind_env", None)):
            self.inner.bind_env(**kw)

    def reset(self):
        self.inner.reset()
        self.x_first = None
        self.x_last = None
        self.states = []

    def step(self, k, t, tau, x, z, f):
        if self.x_first is None:
            self.x_first = np.array(x, copy=True)
            if self.every:
                self.states.append((t, self.x_first))
        x_next, z_next = self.inner.step(k, t, tau, x, z, f)
        self.x_last = np.array(x_next, copy=True)
        if self.every and (k + 1) % self.every == 0:
            self.states.append((t + tau, self.x_last))
        return x_next, z_next

    def describe(self):
        return self.inner.describe()


def simulate_drift(problem, model, disc, tau, **kw):
    """Run ``simulate`` and also return the joint drift ||theta(T) - theta(0)||
    per trajectory (inf for diverged trajectories)."""
    from ..simulate import simulate
    rec = StateRecorder(disc)
    sim = simulate(problem, model, rec, tau, **kw)
    drift = joint_drift(rec.x_last, rec.x_first)
    drift = np.where(sim["diverged"], np.inf, drift)
    return sim, drift


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
from ..registry import register_problem  # noqa: E402


@register_problem("robot_puma560")
def _robot_puma560(path="circle", **kw):
    return ManipulatorTracking(robot="puma560", path=path, **kw)


@register_problem("robot_panda")
def _robot_panda(path="circle", **kw):
    return ManipulatorTracking(robot="panda", path=path, **kw)
