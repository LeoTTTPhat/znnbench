"""Tests for znnbench/problems/robot.py (redundant-manipulator path tracking)."""
import numpy as np
import pytest

from znnbench import make_discretization, make_model, make_problem, simulate
from znnbench.problems import robot as R

PI = np.pi
PATHS = ["circle", "lissajous", "plum", "heart"]
ROBOTS = ["puma560", "panda"]


# ---------------------------------------------------------------------------
# independent reference FK: explicit 4x4 products of elementary transforms
# ---------------------------------------------------------------------------
def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def _tz(d):
    T = np.eye(4)
    T[2, 3] = d
    return T


def _tx(a):
    T = np.eye(4)
    T[0, 3] = a
    return T


def ref_fk(arm, q):
    T = _tz(arm.base_z)
    for i in range(arm.n):
        th = q[i] + arm.offset[i]
        if arm.convention == "standard":
            T = T @ _rz(th) @ _tz(arm.d[i]) @ _tx(arm.a[i]) @ _rx(arm.alpha[i])
        else:
            T = T @ _rx(arm.alpha[i]) @ _tx(arm.a[i]) @ _rz(th) @ _tz(arm.d[i])
    T = T @ _tz(arm.tool_d)
    return T[:3, 3]


def _rand_q(arm, rng, B):
    lo, hi = arm.qlim[:, 0], arm.qlim[:, 1]
    return rng.uniform(lo, hi, size=(B, arm.n))


ARMS = [("puma560", dict()), ("puma560", dict(tool_d=0.0)), ("puma560", dict(dh="jin2017")),
        ("panda", dict()), ("panda", dict(ee="tcp"))]


# ---------------------------------------------------------------------------
# kinematic tables and forward kinematics
# ---------------------------------------------------------------------------
def test_dh_tables_match_sources():
    # Corke, Robotics Toolbox for MATLAB, mdl_puma560.m (standard DH)
    t = R.PUMA560_CORKE
    assert t["convention"] == "standard"
    assert np.allclose(t["a"], [0, 0.4318, 0.0203, 0, 0, 0])
    assert np.allclose(t["d"], [0, 0, 0.15005, 0.4318, 0, 0])
    assert np.allclose(t["alpha"], [PI / 2, 0, -PI / 2, PI / 2, -PI / 2, 0])
    # mdl_puma560.m 'qlim' entries (degrees)
    assert np.allclose(np.rad2deg(t["qlim"]), [[-160, 160], [-45, 225], [-225, 45], [-110, 170],
                                               [-100, 100], [-266, 266]])
    # Jin, Li, La, Luo, IEEE TIE 2017, Table I (re-read from the authors' PDF)
    t = R.PUMA560_JIN2017
    assert np.allclose(t["a"], [0, 0.4318, 0.4318, 0, 0, 0])
    assert np.allclose(t["d"], [0.67, 0, 0.15005, 0, 0, 0.2])
    assert np.allclose(t["alpha"], [PI / 2, 0, -PI / 2, PI / 2, -PI / 2, 0])
    # Franka Control Interface docs (Craig convention) + flange
    t = R.PANDA_FRANKA
    assert t["convention"] == "modified"
    assert np.allclose(t["a"], [0, 0, 0, 0.0825, -0.0825, 0, 0.088])
    assert np.allclose(t["d"], [0.333, 0, 0.316, 0, 0.384, 0, 0])
    assert np.allclose(t["alpha"], [0, -PI / 2, PI / 2, PI / 2, -PI / 2, PI / 2, PI / 2])
    assert t["flange_d"] == 0.107
    assert np.allclose(t["qlim"][:, 1], [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    assert np.allclose(t["qlim"][:, 0], [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,
                                         -2.8973])
    # nominal configurations lie inside the joint limits
    for q, lim in [(R.PUMA560_QN, R.PUMA560_CORKE["qlim"]), (R.PANDA_READY, t["qlim"])]:
        assert np.all(q >= lim[:, 0]) and np.all(q <= lim[:, 1])


def test_fk_reference_values():
    # Corke (Robotics, Vision & Control): p560.fkine(qz) -> t = [0.4521, -0.1500, 0.4318],
    # p560.fkine(qr) -> t = [0.0203, -0.1500, 0.8636]  (no tool, no pedestal)
    puma = R.make_arm("puma560", tool_d=0.0)
    assert np.allclose(puma.fk(np.zeros((1, 6)))[0], [0.4521, -0.15005, 0.4318], atol=1e-9)
    assert np.allclose(puma.fk(np.array([[0, PI / 2, -PI / 2, 0, 0, 0]]))[0],
                       [0.0203, -0.15005, 0.8636], atol=1e-9)
    # Panda at q = 0: flange at (a7, 0, d1 + d3 + d5 - d_flange) = (0.088, 0, 0.926)
    panda = R.make_arm("panda")
    assert np.allclose(panda.fk(np.zeros((1, 7)))[0], [0.088, 0.0, 0.926], atol=1e-12)
    tcp = R.make_arm("panda", ee="tcp")
    assert np.allclose(tcp.fk(np.zeros((1, 7)))[0], [0.088, 0.0, 0.926 - 0.1034], atol=1e-12)


@pytest.mark.parametrize("robot,kw", ARMS)
def test_fk_matches_reference_products(robot, kw):
    arm = R.make_arm(robot, **kw)
    q = _rand_q(arm, np.random.default_rng(0), 5)
    pe = arm.fk(q)
    for b in range(5):
        assert np.allclose(pe[b], ref_fk(arm, q[b]), atol=1e-12)


@pytest.mark.parametrize("robot,kw", ARMS)
def test_jacobian_vs_finite_differences(robot, kw):
    arm = R.make_arm(robot, **kw)
    q = _rand_q(arm, np.random.default_rng(1), 4)
    J = arm.jacobian(q)
    h = 1e-6
    for i in range(arm.n):
        dq = np.zeros(arm.n)
        dq[i] = h
        fd = (arm.fk(q + dq) - arm.fk(q - dq)) / (2 * h)
        assert np.allclose(J[:, :, i], fd, atol=1e-8), (i, J[:, :, i], fd)


def test_puma_tool_offset_gives_redundancy():
    q = R.PUMA560_QN[None]
    no_tool = R.make_arm("puma560", tool_d=0.0).jacobian(q)[0]
    assert np.allclose(no_tool[:, 3:], 0.0, atol=1e-12)      # wrist centre: joints 4-6 idle
    tool = R.make_arm("puma560").jacobian(q)[0]
    assert np.linalg.norm(tool[:, 3:5]) > 0.1                 # d6 = 0.2 m tool moves j4, j5


# Verification addition: the tool / flange offset lies on the last joint axis
# (z6 = z5 for the standard-DH PUMA with alpha6 = 0, z7 for the Panda), so the
# last joint never moves the end-effector point -> effective redundancy n - 4.
@pytest.mark.parametrize("robot,kw", ARMS)
def test_last_joint_column_vanishes(robot, kw):
    arm = R.make_arm(robot, **kw)
    q = _rand_q(arm, np.random.default_rng(7), 20)
    J = arm.jacobian(q)
    assert np.abs(J[:, :, -1]).max() < 1e-14
    assert np.all(np.linalg.matrix_rank(J[:, :, :-1]) == 3)    # rank 3 from the others


# Verification addition: independent kinematic source for the Panda -- the URDF
# joint origins of franka_description/robots/common/franka_arm.xacro
# (child = parent * Trans(xyz) * Rx(roll) * Rz(q); joint8 fixed at z = 0.107).
_PANDA_URDF = [((0, 0, 0.333), 0.0), ((0, 0, 0), -PI / 2), ((0, -0.316, 0), PI / 2),
               ((0.0825, 0, 0), PI / 2), ((-0.0825, 0.384, 0), -PI / 2), ((0, 0, 0), PI / 2),
               ((0.088, 0, 0), PI / 2)]


def _panda_urdf_fk(q, extra=0.107):
    T = np.eye(4)
    for (xyz, roll), qi in zip(_PANDA_URDF, q):
        Tt = np.eye(4)
        Tt[:3, 3] = xyz
        T = T @ Tt @ _rx(roll) @ _rz(qi)
    return (T @ _tz(extra))[:3, 3]


def test_panda_matches_franka_urdf_and_ready_pose():
    arm, tcp = R.make_arm("panda"), R.make_arm("panda", ee="tcp")
    q = _rand_q(arm, np.random.default_rng(8), 10)
    pe, pt = arm.fk(q), tcp.fk(q)
    for b in range(10):
        assert np.allclose(pe[b], _panda_urdf_fk(q[b]), atol=1e-12)
        assert np.allclose(pt[b], _panda_urdf_fk(q[b], 0.107 + 0.1034), atol=1e-12)
    # libfranka "ready" pose: flange (0.3069, 0, 0.5903), hand TCP (0.3069, 0, 0.4869)
    assert np.allclose(arm.fk(R.PANDA_READY[None])[0], [0.30689, 0.0, 0.59028], atol=1e-5)
    assert np.allclose(tcp.fk(R.PANDA_READY[None])[0], [0.30689, 0.0, 0.48688], atol=1e-5)


def test_puma_pedestal_and_named_poses():
    # Python toolbox model = MATLAB model + 26.45-in pedestal (same Jacobian)
    bare = R.make_arm("puma560", tool_d=0.0)
    ped = R.make_arm("puma560", tool_d=0.0, base_z=R.PUMA560_PEDESTAL)
    q = _rand_q(bare, np.random.default_rng(9), 4)
    assert np.isclose(R.PUMA560_PEDESTAL, 0.67183)
    assert np.allclose(ped.fk(q) - bare.fk(q), [0, 0, 0.67183], atol=1e-12)
    assert np.allclose(ped.jacobian(q), bare.jacobian(q), atol=1e-15)
    # qs (straight, horizontal) and qn: hand-derivable wrist-centre positions
    assert np.allclose(bare.fk(np.array([[0, 0, -PI / 2, 0, 0, 0]]))[0],
                       [0.8636, -0.15005, -0.0203], atol=1e-12)
    # qn: upper arm a2 at +45 deg, a3 along -135 deg, forearm d4 along -45 deg
    s = np.sqrt(0.5)
    assert np.allclose(bare.fk(R.PUMA560_QN[None])[0],
                       [(0.4318 - 0.0203 + 0.4318) * s, -0.15005,
                        (0.4318 - 0.0203 - 0.4318) * s], atol=1e-12)
    # +pi (as listed by Corke) and -pi (used here) are the same pose
    qn_plus = R.PUMA560_QN.copy()
    qn_plus[2] = PI
    assert np.allclose(R.make_arm("puma560").fk(qn_plus[None]),
                       R.make_arm("puma560").fk(R.PUMA560_QN[None]), atol=1e-12)


# ---------------------------------------------------------------------------
# problem API
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("robot", ROBOTS)
def test_registry_and_shapes(robot):
    P = make_problem(f"robot_{robot}")
    n = 6 if robot == "puma560" else 7
    assert (P.n_x, P.m, P.family) == (n, 3, "robot")
    assert P.x_star(0.3) is None
    x = P.x0(np.random.default_rng(0), 3)
    assert x.shape == (3, n) and np.allclose(x, P.theta0)
    d = P.data(0.5)
    assert P.E(x, d).shape == (3, 3) and P.dEdt(x, 0.5).shape == (3, 3)
    assert P.solve_J(x, d, np.ones((3, 3))).shape == (3, n)
    assert P.JT(x, d, np.ones((3, 3))).shape == (3, n)
    assert "robot_kinematics" in P.capabilities()


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("path", PATHS)
def test_path_anchor_closure_and_dEdt(robot, path):
    P = make_problem(f"robot_{robot}", path=path)
    th0 = P.theta0[None]
    assert np.abs(P.E(th0, P.data(0.0))).max() < 1e-14           # starts on the path
    assert np.allclose(P.rd(P.T), P.rd(0.0), atol=1e-12)          # closed after one period
    x = P.theta0 + np.random.default_rng(2).uniform(-0.2, 0.2, size=(2, P.n_x))
    h = 1e-6
    for t in [0.0, 1.3, 4.7, 9.9]:
        fd = (P.E(x, P.data(t + h)) - P.E(x, P.data(t - h))) / (2 * h)
        assert np.allclose(fd, P.dEdt(x, t), atol=1e-8)
        fdr = (P.rd(t + h) - P.rd(t - h)) / (2 * h)
        assert np.allclose(fdr, P.rd_dot(t), atol=1e-8)


def test_smooth_phase_zero_end_speeds():
    for robot in ROBOTS:
        for path in ["circle", "plum"]:
            P = make_problem(f"robot_{robot}", path=path)
            assert np.abs(P.rd_dot(0.0)).max() < 1e-14
            assert np.abs(P.rd_dot(P.T)).max() < 1e-12


def _dense_J(P, x, d, h=1e-6):
    B, n = x.shape
    J = np.empty((B, P.m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = h
        J[:, :, i] = (P.E(x + dx, d) - P.E(x - dx, d)) / (2 * h)
    return J


@pytest.mark.parametrize("robot", ROBOTS)
def test_solve_J_is_min_norm_pinv_and_JT(robot):
    P = make_problem(f"robot_{robot}", path="lissajous")
    rng = np.random.default_rng(3)
    x = P.theta0 + rng.uniform(-0.3, 0.3, size=(4, P.n_x))
    d = P.data(2.0)
    r = rng.standard_normal((4, 3))
    Jfd = _dense_J(P, x, d)
    dx = P.solve_J(x, d, r)
    for b in range(4):
        assert np.allclose(Jfd[b] @ dx[b], r[b], atol=1e-7)          # J dx = r
        assert np.allclose(dx[b], np.linalg.pinv(Jfd[b]) @ r[b], atol=1e-6)  # min norm
    v = rng.standard_normal((4, 3))
    assert np.allclose(P.JT(x, d, v), np.einsum("bmn,bm->bn", Jfd, v), atol=1e-7)
    # damped pseudoinverse J^T (J J^T + lambda^2 I)^{-1} r
    Pd = make_problem(f"robot_{robot}", path="lissajous", damping=0.05)
    J = Pd.jacobian(x)
    ref = np.stack([J[b].T @ np.linalg.solve(J[b] @ J[b].T + 0.05 ** 2 * np.eye(3), r[b])
                    for b in range(4)])
    assert np.allclose(Pd.solve_J(x, d, r), ref, atol=1e-12)


def test_solve_J_singular_fallback():
    # A planar arm (all alpha = 0) has an exactly zero z-row in J, so J J^T is
    # exactly singular and solve_J must fall back to the least-squares pinv.
    P = make_problem("robot_puma560")
    P.arm = R.SerialArm("planar3", "standard", a=[0.4, 0.3, 0.2], d=[0, 0, 0], alpha=[0, 0, 0],
                        qlim=np.array([[-PI, PI]] * 3))
    P.n_x, P._ck = 3, None
    x = np.array([[0.3, -0.5, 0.8], [0.1, 0.2, 0.3]])
    r = np.array([[0.01, 0.02, -0.03], [0.0, 0.1, 0.2]])
    J = P.jacobian(x)
    assert np.all(J[:, 2, :] == 0.0)
    dx = P.solve_J(x, P.data(0.0), r)
    for b in range(2):
        assert np.allclose(dx[b], np.linalg.pinv(J[b]) @ r[b], atol=1e-12)
    # near (not exactly) singular configurations: damping keeps theta_dot bounded
    Pd = make_problem("robot_puma560", tool_d=0.0, damping=1e-2)
    xs = np.array([[0.0, 0.0, -PI / 2 + np.arctan2(0.0203, 0.4318), 0.0, 0.0, 0.0]])
    sv = np.linalg.svd(Pd.jacobian(xs)[0], compute_uv=False)
    assert sv[-1] < 1e-9                                            # stretched elbow
    assert np.linalg.norm(Pd.solve_J(xs, Pd.data(0.0), np.ones((1, 3)))) < 1e3


def test_batch_consistency_and_cache():
    P = make_problem("robot_panda", path="plum")
    rng = np.random.default_rng(4)
    x = P.theta0 + rng.uniform(-0.3, 0.3, size=(3, P.n_x))
    d = P.data(3.0)
    r = rng.standard_normal((3, 3))
    Eb, Sb, Tb = P.E(x, d), P.solve_J(x, d, r), P.JT(x, d, r)
    for b in range(3):
        xb = x[b:b + 1]
        assert np.allclose(P.E(xb, d), Eb[b:b + 1], atol=1e-14)
        assert np.allclose(P.solve_J(xb, d, r[b:b + 1]), Sb[b:b + 1], atol=1e-12)
        assert np.allclose(P.JT(xb, d, r[b:b + 1]), Tb[b:b + 1], atol=1e-14)
    # cache must not return stale values after an in-place change of x
    x2 = x.copy()
    e1 = P.E(x2, d)
    x2[0, 0] += 0.1
    assert not np.allclose(P.E(x2, d)[0], e1[0])


# ---------------------------------------------------------------------------
# dynamics smoke tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("path", PATHS)
def test_znn_euler_tracking(robot, path):
    P = make_problem(f"robot_{robot}", path=path)
    s = simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-3, gamma=10.0, B=1)
    assert not s["diverged"].any()
    assert np.max(s["residual"]) < 1e-3, np.max(s["residual"])


def test_znn_converges_from_off_path_start_and_gnn_lags():
    P = make_problem("robot_puma560", path="circle", x0_spread=0.05, T=2.0, path_T=10.0)
    s = simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-3, gamma=10.0,
                 B=2, seed=1)
    r = s["residual"]
    assert np.all(r[0] > 1e-3)                              # starts off the path
    # exponential decay at rate gamma down to the O(tau) Euler tracking floor
    assert np.all(r[500] < 0.05 * r[0]) and np.all(r[-1] < 1e-4)
    # GNN (no feed-forward) keeps an O(1/(gamma sigma_min^2)) lag behind the moving
    # target (sigma_min(J) ~ 0.26 at qn; gamma tau sigma_max^2 < 2 keeps Euler stable)
    Pg = make_problem("robot_puma560", path="lissajous", T=3.0)
    lag = [np.max(simulate(Pg, make_model("gnn"), make_discretization("euler"), tau=1e-3,
                           gamma=g)["residual"][-500:]) for g in (100.0, 1000.0)]
    assert lag[1] < lag[0] / 3 and lag[0] > 1e-3


def test_backward_difference_information_regime():
    P = make_problem("robot_panda", path="lissajous", T=3.0)
    s = simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-3, gamma=10.0,
                 info="bd1")
    assert np.max(s["residual"]) < 1e-3


def test_joint_drift_recorder_and_limits():
    P = make_problem("robot_panda", path="circle")
    rec = R.StateRecorder(make_discretization("euler"), every=250)
    s = simulate(P, make_model("znn"), rec, tau=1e-3, gamma=10.0, B=1)
    assert np.allclose(rec.x_first, P.theta0[None])
    # the recorded final state reproduces the final residual of simulate
    assert np.isclose(P.residual(rec.x_last, P.data(P.T))[0], s["residual"][-1, 0], rtol=1e-12)
    drift = R.joint_drift(rec.x_last, rec.x_first)
    assert drift.shape == (1,) and np.isfinite(drift).all()
    pj = R.joint_drift(rec.x_last, rec.x_first, per_joint=True)
    assert np.isclose(np.linalg.norm(pj), drift[0])
    assert np.all(R.joint_drift(P.theta0, P.theta0) == 0)
    s2, drift2 = R.simulate_drift(P, make_model("znn"), make_discretization("euler"), 1e-3,
                                  gamma=10.0)
    assert np.allclose(drift2, drift)
    # default paths stay inside the joint limits of both arms
    th = np.concatenate([x for _, x in rec.states])
    assert R.qlim_violation(th, P.arm.qlim).max() == 0.0
    assert R.qlim_violation(np.array([[0, 2.0, 0, -1, 0, 1, 0]]), P.arm.qlim)[0] > 0.2


def test_continuous_time_error_decays_exactly_exponentially():
    # With full-row-rank J, J theta_dot = rd_dot - gamma E gives dE/dt = -gamma E
    # exactly (Jin et al. 2017, text after eq. (12)); RK4 with gamma tau = 5e-3
    # reproduces E(t) = E(0) exp(-gamma t) to ~1e-7 relative accuracy.
    P = make_problem("robot_panda", path="plum", x0_spread=0.05, T=1.0)
    g = 5.0
    s = simulate(P, make_model("znn"), make_discretization("rk4"), tau=1e-3, gamma=g, B=2,
                 seed=3)
    ratio = s["residual"] / s["residual"][0]
    assert np.all(s["residual"][0] > 1e-2)
    assert np.max(np.abs(ratio / np.exp(-g * s["t"])[:, None] - 1.0)) < 1e-5


def test_euler_order_at_fixed_h():
    # Forward-Euler ZNN at fixed h = gamma tau: steady residual O(tau^2) (DESIGN.md
    # order check); at fixed gamma: O(tau).
    from znnbench.metrics import empirical_order, mssre
    taus = [4e-3, 2e-3, 1e-3]
    eh, eg = [], []
    for tau in taus:
        P = make_problem("robot_puma560", path="lissajous", T=3.0)
        eh.append(mssre(simulate(P, make_model("znn"), make_discretization("euler"), tau=tau,
                                 gamma=0.1 / tau))[0])
        eg.append(mssre(simulate(P, make_model("znn"), make_discretization("euler"), tau=tau,
                                 gamma=10.0))[0])
    assert abs(empirical_order(taus, eh)[0] - 2.0) < 0.15
    assert abs(empirical_order(taus, eg)[0] - 1.0) < 0.15


def test_absolute_centre_jin2017_circle():
    # Jin, Li, La, Luo (TIE 2017) Sec. VI-B: centre [0.25, 0, 1.3] m, radius 0.2 m,
    # plane revolved 30 deg about X, 0.2 rad/s, k0 = 5, start off the path.
    w = 0.2
    P = make_problem("robot_puma560", path="circle", dh="jin2017", T=4.0, path_T=2 * PI / w,
                     center=(0.25, 0.0, 1.3), path_kw=dict(radius=0.2, phase="linear"))
    assert P.name == "robot_puma560_circle_jin2017_abs"
    c, z = np.array([0.25, 0.0, 1.3]), PI / 6
    for t in [0.0, 1.7, 9.0, 25.0]:
        ref = c + 0.2 * np.array([np.cos(w * t), np.sin(w * t) * np.cos(z),
                                  np.sin(w * t) * np.sin(z)])
        assert np.allclose(P.rd(t), ref, atol=1e-14)
        assert np.allclose(P.rd_dot(t), 0.2 * w * np.array([-np.sin(w * t), np.cos(w * t) * np.cos(z),
                                                            np.cos(w * t) * np.sin(z)]),
                           atol=1e-14)
    s = simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-3, gamma=5.0)
    r = s["residual"][:, 0]
    assert r[0] > 0.3                                        # initial position off the path
    assert r[-1] < 1e-4 and np.all(np.diff(r[:2000]) < 0)     # monotone exponential decay
    assert make_problem("robot_puma560", path="circle").center is None


def test_jin2017_variant_tracks():
    P = make_problem("robot_puma560", path="circle", dh="jin2017")
    assert P.name == "robot_puma560_circle_jin2017"
    lim = P.arm.qlim
    assert np.all(P.theta0 >= lim[:, 0]) and np.all(P.theta0 <= lim[:, 1])
    assert np.linalg.svd(P.jacobian(P.theta0)[0], compute_uv=False)[-1] > 0.2
    s = simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-3, gamma=10.0)
    assert np.max(s["residual"]) < 1e-3
