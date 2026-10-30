# Spec: redundant-manipulator path tracking (`problems_robot`)

Files: `znnbench/problems/robot.py`, `tests/test_problems_robot.py`, this document.

Registered keys: `robot_puma560` (6 joints) and `robot_panda` (7 joints). Both factories take
`path=` with one of `circle`, `lissajous`, `plum` or `heart`, plus the keyword options of
`ManipulatorTracking` listed in Sec. 5.

## 0. Problem definition

The application is the velocity-level redundancy resolution used in most ZNN/GNN robotics
papers, for example Jin, Li, La, Luo (TIE 2017) and Li, Zhang, Jin (TNNLS 2017). The task is
position only, so the end-effector position $r=f(\theta)\in\mathbb R^3$ follows a desired path
$r_d(t)$. Nominally a 6-joint arm then has 3 redundant DOF and a 7-joint arm has 4.

**Verified correction: the effective redundancy is $n-4$.** On both benchmark arms the
end-effector point lies on the axis of the last joint, so the last column of the position
Jacobian is identically zero (measured: below 1.1e-16 at random configurations). For the
standard-DH PUMA, $\alpha_6=0$ gives $z_6=z_5$, and the tool is along $z_6$. For the Panda, the
flange and TCP are along $z_7$. The PUMA therefore has 2 effective redundant DOF and the Panda
has 3; the last joint receives zero pseudoinverse velocity. The same holds for the PUMA table
of Jin et al. (2017): its $d_6=0.2$ m is a translation along $z_5$. The paper nevertheless
treats the arm as having 3 redundant DOF for the position task. The test
`test_last_joint_column_vanishes` checks this for all 5 arm variants.

$$E(\theta,t)=f(\theta)-r_d(t)\in\mathbb R^{3},\qquad
J(\theta)=\frac{\partial f}{\partial\theta}\in\mathbb R^{3\times n},\qquad
\frac{\partial E}{\partial t}\Big|_{\theta}=-\dot r_d(t).$$

With the ZNN design formula $\dot E=-\gamma\Phi(E)$, the model becomes

$$J(\theta)\dot\theta=\dot r_d(t)-\gamma\,\Phi\big(f(\theta)-r_d(t)\big),
\qquad \dot\theta=J^{+}\big(\dot r_d-\gamma\Phi(f(\theta)-r_d)\big).$$

With the linear activation this is exactly eqs. (12)–(13) of Jin et al. (2017),
$Jw=-k_0(f(\theta)-r_d)+\dot r_d$ with $w=\dot\theta$ and $k_0=\gamma$, as read from the full
text. In ZNN-Bench, `models/znn.py` supplies the design formula. The problem supplies the
following methods:

| method | implementation |
|---|---|
| `data(t)` | `{'t', 'rd'}`, where `rd` $=p_0+\text{path}(t)$, shape (3,) |
| `E(x,d)` | $f(\theta)-r_d$, shape (B, 3). FK is vectorised over the batch. |
| `dEdt(x,t)` | $-\dot r_d(t)$, broadcast to (B, 3) and exact |
| `solve_J(x,d,r)` | $J^T(JJ^T+\lambda^2I)^{-1}r$ with `damping` $=\lambda$ (default 0, the exact min-norm pseudoinverse). If $JJ^T$ is exactly singular (`LinAlgError`), it falls back to `pinv(J) r`. |
| `JT(x,d,v)` | $J^Tv$ (for GNN) |
| `x_star` | `None`, because the problem is redundant. The residual $\|E\|_2$ is the Cartesian position error in metres. |
| `x0(rng,B)` | $\theta_0$, plus $U[-s,s]$ per joint when `x0_spread`$=s>0$ |
| `capabilities()` | `{'robot_kinematics', 'redundant'}`. The problem also exposes `fk`, `jacobian`, `manipulability` ($\sqrt{\det JJ^T}$, as in Jin et al. 2017, eq. (9)), `theta0` and `arm.qlim`. |

FK and the Jacobian share a one-entry cache keyed by the **values** of x (an O(Bn) equality
test). As a result, `E`, `solve_J`, `JT` and the recorded residual at the same sampling instant
cost one FK evaluation. A test checks that the cache cannot go stale after an in-place change
of x.

## 1. Kinematics (batched, `SerialArm`)

The arms are all-revolute. Link transforms are written as products of elementary transforms:

* Standard DH (Corke `RevoluteDH`): $A_i=R_z(\theta_i)T_z(d_i)T_x(a_i)R_x(\alpha_i)$. Joint i
  rotates about $z_{i-1}$.
* Modified (Craig) DH (Corke `RevoluteMDH`, Franka docs):
  $A_i=R_x(\alpha_{i-1})T_x(a_{i-1})R_z(\theta_i)T_z(d_i)$. Joint i rotates about $z_i$.
* The end-effector is $p_e=$ origin of the last frame $+\,d_{tool}\,z_{last}$.
* Each column of the geometric position Jacobian is $J_{:,i}=z_{\text{axis}(i)}\times(p_e-o_{\text{axis}(i)})$.

Only rotations (B,3,3) and positions (B,3) are propagated, with one loop over joints and no
loop over the batch.

### 1a. `robot_puma560`, default `dh="corke"`

**Source.** P. Corke, *Robotics Toolbox for MATLAB*, `models/mdl_puma560.m`. The file was
fetched from github.com/petercorke/robotics-toolbox-matlab and its values transcribed. They are
cross-checked against the Python toolbox `src/roboticstoolbox/models/DH/Puma560.py`, also
fetched. The book is P. Corke, *Robotics, Vision and Control*, 2nd ed., Springer 2017,
DOI [10.1007/978-3-319-54413-7](https://doi.org/10.1007/978-3-319-54413-7). The parameter
consensus is P. Corke, B. Armstrong-Hélouvry, "A search for consensus among model parameters
reported for the PUMA 560 robot", ICRA 1994, pp. 1608–1613,
DOI [10.1109/ROBOT.1994.351360](https://doi.org/10.1109/ROBOT.1994.351360). The Python toolbox
paper is P. Corke, J. Haviland, ICRA 2021,
DOI [10.1109/ICRA48506.2021.9561366](https://doi.org/10.1109/ICRA48506.2021.9561366).

| link | $a$ (m) | $d$ (m) | $\alpha$ (rad) | limits (deg, `qlim` of `mdl_puma560.m`) |
|---|---|---|---|---|
| 1 | 0 | 0 | $\pi/2$ | [-160, 160] |
| 2 | 0.4318 | 0 | 0 | [-45, 225] |
| 3 | 0.0203 | 0.15005 | $-\pi/2$ | [-225, 45] |
| 4 | 0 | 0.4318 | $\pi/2$ | [-110, 170] |
| 5 | 0 | 0 | $-\pi/2$ | [-100, 100] |
| 6 | 0 | 0 | 0 | [-266, 266] |

* The two toolboxes have identical link parameters but differ in two places.
  * **Pedestal.** The Python toolbox sets $d_1=26.45$ in $=0.67183$ m. Its docstring warns
    that, compared with the MATLAB version, the model "includes the pedestal". We use
    $d_1=0$, the MATLAB value. The option `base_z=PUMA560_PEDESTAL` (0.67183 m) reproduces
    the Python model's positions, and the pedestal does not change $J$ (tested). The earlier
    justification, that "MATLAB-based ZNN papers use $d_1=0$", was removed: Jin et al. (2017)
    use $d_1=0.67$.
  * **Joint limits.** The limits above are the `qlim` entries of `mdl_puma560.m`. The Python
    `Puma560.py` keeps them only as comments. Its active limits are [-160,160], [-110,110],
    [-135,135], [-266,266], [-100,100] and [-266,266] deg, and it cites Wang & Chen, IEEE
    T-RA 7(4), 1991, for them. (The previous version of this spec wrongly attributed the
    MATLAB values to the Python toolbox.) Under the Python limits, $\theta_3=\pm\pi$ in `qn`
    is out of range.
* **ASSUMPTION, tool offset.** The table has $d_6=0$, so the end-effector is the wrist centre.
  The three wrist axes then pass through it, and the last three Jacobian columns are
  identically zero. The test `test_puma_tool_offset_gives_redundancy` checks this. Without a
  tool the "3 redundant DOF" are degenerate. The default `tool_d=0.2` m along $z_6$ is the
  $d_6$ of Jin et al. (2017), Table I. Set `tool_d=0` to get the bare Corke arm.
* **Nominal configuration.** Corke's `qn` ("nominal table top picking pose") is used. Both
  `mdl_puma560.m` and the Python `Puma560.py` list `[0 pi/4 pi 0 pi/4 0]`. (The previous
  version of this spec claimed that the Python toolbox uses $-\pi$; that claim was false and
  has been corrected.) **ASSUMPTION (angle wrapping only):** we use $\theta_3=-\pi$. It is
  the same pose (tested), and unlike $+\pi$ it lies inside the MATLAB joint-3 limit
  [-225°, 45°]. With the tool, $p_0=f(\theta_0)=(0.7963,-0.1501,-0.0144)$ m and
  $\sigma_{\min}(J)=0.26$. Without the tool, the wrist centre is at
  $((a_2-a_3+d_4)/\sqrt2,\,-d_3,\,(a_2-a_3-d_4)/\sqrt2)=(0.5963,-0.15005,-0.0144)$, which is
  derived by hand and tested.

### 1b. `robot_puma560` with `dh="jin2017"` (the PUMA table used in ZNN papers)

**Source.** L. Jin, S. Li, H. M. La, X. Luo, "Manipulability optimization of redundant
manipulators using dynamic neural networks," *IEEE Trans. Ind. Electron.* 64(6):4710–4720,
2017, DOI [10.1109/TIE.2017.2674624](https://doi.org/10.1109/TIE.2017.2674624). Table I was
read from the authors' PDF hosted at ara.cse.unr.edu.

| link | $a$ (m) | $\alpha$ (rad) | $d$ (m) |
|---|---|---|---|
| 1 | 0 | $\pi/2$ | 0.67 |
| 2 | 0.4318 | 0 | 0 |
| 3 | 0.4318 | $-\pi/2$ | 0.15005 |
| 4 | 0 | $\pi/2$ | 0 |
| 5 | 0 | $-\pi/2$ | 0 |
| 6 | 0 | 0 | 0.2 |

**Finding.** This table is *not* the Corke/Armstrong standard-DH PUMA:

* The forearm length is placed in $a_3$ instead of $d_4$, and $a_3=0.0203$ is dropped. Joint 4
  is therefore perpendicular to the forearm rather than along it.
* $d_1=0.67$ m matches the 26.45-in pedestal.
* $d_6=0.2$ m is a tool.

* In standard DH, $d_6$ is a translation along $z_5$, the joint-6 axis. Joint 6 therefore
  never moves the end-effector point (see Sec. 0).

`dh="jin2017"` reproduces the table verbatim. **ASSUMPTION:** the paper starts from random
configurations. Our nominal $\theta_0$ is Corke's `qn` with $\theta_3+\pi/2$, which compensates
the $\pi/2$ rotation of the forearm direction ($x_3$ vs $z_3$). That gives
$\sigma_{\min}(J)=0.26$ and $p_0=(0.611,-0.150,0.870)$.

### 1c. `robot_panda` (Franka Emika Panda / FER)

**Source.** Franka Robotics, *Franka Control Interface documentation*, page "Control Interface
Specification and Robot Limits", section "Denavit–Hartenberg Parameters" (Craig's convention).
The page was fetched from frankarobotics.github.io/docs/robot_specifications.html; the old URL
`frankaemika.github.io/docs/control_parameters.html` now returns 404. The page labels the
table "Franka Research 3 kinematic chain". It is numerically identical to the FER/Panda table
in Corke's Python toolbox `models/DH/Panda.py`, which cites the old Franka URL and was fetched
and compared. The Panda reference paper is S. Haddadin et al., "The Franka Emika Robot: A
reference platform for robotics research and education," *IEEE Robot. Autom. Mag.*
29(2):46–64, 2022, DOI [10.1109/MRA.2021.3138382](https://doi.org/10.1109/MRA.2021.3138382).

| joint | $a$ (m) | $d$ (m) | $\alpha$ (rad) | $q_{\min}$ | $q_{\max}$ (rad, FER limits, same page) |
|---|---|---|---|---|---|
| 1 | 0 | 0.333 | 0 | -2.8973 | 2.8973 |
| 2 | 0 | 0 | $-\pi/2$ | -1.7628 | 1.7628 |
| 3 | 0 | 0.316 | $\pi/2$ | -2.8973 | 2.8973 |
| 4 | 0.0825 | 0 | $\pi/2$ | -3.0718 | -0.0698 |
| 5 | -0.0825 | 0.384 | $-\pi/2$ | -2.8973 | 2.8973 |
| 6 | 0 | 0 | $\pi/2$ | -0.0175 | 3.7525 |
| 7 | 0.088 | 0 | $\pi/2$ | -2.8973 | 2.8973 |
| flange | 0 | 0.107 | 0 | | |

* The end-effector is the flange by default. With `ee="tcp"` it is the Franka-hand TCP, a
  further +0.1034 m along the flange z axis (`franka_description/robots/common/franka_hand.xacro`,
  `tcp_xyz='0 0 0.1034'`, fetched). Corke's DH model uses 0.103 m instead.
* **Nominal configuration:** the libfranka example "ready" pose
  $q=(0,-\pi/4,0,-3\pi/4,0,\pi/2,\pi/4)$ (`examples/generate_joint_position_motion.cpp`,
  `q_goal`, fetched). It gives $p_0=(0.3069,0,0.5903)$ m at the flange and
  $(0.3069,0,0.4869)$ m at the TCP, with $\sigma_{\min}(J)=0.26$ at the flange.
* **Independent cross-check (verification).** The URDF joint origins in
  `franka_ros/franka_description/robots/common/franka_arm.xacro` were fetched. They are
  joint1 `xyz 0 0 0.333`; joint2 `rpy -pi/2`; joint3 `xyz 0 -0.316 0, rpy pi/2`; joint4
  `xyz 0.0825 0 0, rpy pi/2`; joint5 `xyz -0.0825 0.384 0, rpy -pi/2`; joint6 `rpy pi/2`;
  joint7 `xyz 0.088 0 0, rpy pi/2`; and the fixed joint8 `xyz 0 0 0.107`. This file is the
  common arm macro that `panda.urdf.xacro` (arm_id `panda`) instantiates, so FER/Panda and
  FR3 share these kinematics and differ only in joint limits. That settles the open "FR3
  label" question. The URDF chain $T\,\mathrm{Trans}(xyz)\,R_x(\text{roll})\,R_z(q)$ reproduces
  the DH FK to 1e-12 at random configurations (tested), and the ready-pose TCP is at
  (0.3069, 0, 0.4869).

## 2. Desired paths

Each path is a planar curve $(u(t),v(t))$ in the plane spanned by $e_u=X$ and
$e_v=R_x(\text{tilt})Y$. It is shifted to pass through $p_0=f(\theta_0)$ at $t=0$:

$$r_d(t)=p_0+(u(t)-u(0))e_u+(v(t)-v(0))e_v,\qquad \dot r_d=\dot u\,e_u+\dot v\,e_v,$$

with exact analytic derivatives. **ASSUMPTION, shared by all paths (default):** the source
papers give paths about absolute centres. By default we anchor each path at the nominal start
instead, so that $E(\theta_0,0)=0$. The option `center=c` (added during verification) gives
the source-paper form $r_d(t)=c+u(t)e_u+v(t)e_v$. The start $\theta_0$ is then generally off
the path, and the problem name gets the suffix `_abs`. The exact Jin et al. (2017, Sec. VI-B)
circle task is

```python
make_problem("robot_puma560", path="circle", dh="jin2017", center=(0.25, 0, 1.3),
             path_T=2*np.pi/0.2, path_kw=dict(radius=0.2, phase="linear"), T=...)
# run with gamma = k0 = 5
```

With this setting, ZNN-Euler at τ=1e-3 converges from 0.48 m initial error to 5e-7 m and
violates no joint limit (measured over one 31.4 s period, starting from the jin2017 nominal
pose). The test `test_absolute_centre_jin2017_circle` covers it.

There are two phase laws, and the path period defaults to the horizon T (10 s):

* `smooth`: $\phi(t)=2\pi\sin^2\!\big(\tfrac{\pi t}{2T}\big)$ and
  $\dot\phi=\tfrac{\pi^2}{T}\sin\tfrac{\pi t}{T}$. This is Yang et al. (2025), eq. (31),
  written there as $2\pi\sin(\pi t/2/T)\sin(\pi t/2/T)$, and it is the standard choice in
  Zhang's repetitive-motion work. The path closes at T with zero speed at $t=0$ and $t=T$. For
  $t>T$ it retraces backwards.
* `linear`: $\phi=2\pi t/T$.

| key | curve | defaults | source |
|---|---|---|---|
| `circle` | $u=R\cos\phi$, $v=R\sin\phi$, so $r_d=c+R[\cos\phi,\ \sin\phi\cos\zeta,\ \sin\phi\sin\zeta]$ | $R=0.1$ m, tilt $\zeta=\pi/6$ (30° about X), smooth | Jin et al. 2017, Sec. VI-B: centre $c=[0.25,0,1.3]$ m, radius 0.2 m, plane revolved 30° about X, 0.2 rad/s, $k_0=5$, joint-velocity box $\Omega=[-0.5,0.5]^6$ (re-read from the PDF). **ASSUMPTION:** the rotation sign (+30°) and $\phi(0)=0$ are not stated in the paper. |
| `lissajous` | $x=A\sin(2\pi a t/T+\delta)$, $y=B\sin(2\pi b t/T)$, $z=z_0$ | $A=B=0.06$ m, $a=3$, $b=2$, $\delta=\pi/2$, linear | Yang et al. 2025, eq. (30) (Franka Panda simulation) |
| `plum` | $x=r(\cos\phi+\cos(n\phi)/n)$, $y=r(\sin\phi+\sin(n\phi)/n)$, $z=z_0$ | $r=0.1$ m, $n=5$, smooth | Yang et al. 2025, eq. (31), "five-petal plum" (rose-type) |
| `heart` | $x=a(2\sin\theta-\sin2\theta)$, $y=a(2\cos\theta-\cos2\theta)-a$, $\theta=2\pi t/T$ | $a=0.03$ m, linear | Yang et al. 2025, eq. (29) |

The second source is Y. Yang, X. Li, X. Wang, M. Liu, J. Yin, W. Li, R. M. Voyles, X. Ma, "A
strictly predefined-time convergent and anti-noise fractional-order zeroing neural network for
solving time-variant quadratic programming in kinematic robot control," *Neural Networks*
186:107279, 2025, DOI [10.1016/j.neunet.2025.107279](https://doi.org/10.1016/j.neunet.2025.107279)
(arXiv:2503.01857, full text read).

Assumptions and deviations:

* **ASSUMPTION, `circle` radius 0.1 m (paper: 0.2 m).** A 0.2 m circle anchored at the
  nominal start reaches 0.4 m towards the base. Under pseudoinverse tracking it pushes PUMA
  joint 3 below −225° and Panda joint 2 below −1.7628 rad, which was measured. Pass
  `path_kw=dict(radius=0.2)` for the paper value. The paper's constant angular speed
  of 0.2 rad/s corresponds to `path_kw=dict(phase="linear")` with `path_T=2*pi/0.2`.
* **ASSUMPTION, `heart` scale `a = 0.03` m.** Yang et al. leave the scaling factor $a$
  unspecified.
* Yang et al. use these paths with the QP of their eq. (27):
  $\min\tfrac12\dot q^T\dot q+\rho^T\dot q$ s.t. $J\dot q=\dot w$, $q^-\le q\le q^+$,
  $\dot q^-\le\dot q\le\dot q^+$, where $\rho=\iota(q-q_0)$ is a repetitive-motion term with
  $\iota=1$. Joint limits are folded into velocity bounds by eq. (28), and
  $\dot q^\pm=\pm0.65$ rad/s (re-read during verification). Two differences matter. First,
  their equality constraint has **no position-error feedback** term. Second, they use
  $\rho\neq0$. Without bounds and with $\rho=0$, the QP solution is the min-norm pseudoinverse
  solution used here, and our design formula adds the feedback $-\gamma\Phi(E)$ of Jin et al.
  (2017), eq. (12). The plain problem therefore matches neither paper's full scheme; it is the
  common core of both.
* **Not implemented: "butterfly" and "rhodonea rose" paths.** No ZNN paper with an explicit,
  transcribable butterfly or rose formula was found during this work; the IE-RNN paper
  (arXiv:2008.08228), for example, names a butterfly path but gives no equation. They were left
  out rather than invented. The `plum` curve is the rose-type petal path.

## 3. Repetitive-motion metric and helpers

* `joint_drift(theta_T, theta_0, per_joint=False)` returns $\|\theta(T)-\theta(0)\|_2$ per
  trajectory, or the per-joint drift. This is the repetitive-motion criterion of Zhang & Zhang,
  *Repetitive Motion Planning and Control of Redundant Robot Manipulators*, Springer 2013,
  DOI [10.1007/978-3-642-37518-7](https://doi.org/10.1007/978-3-642-37518-7). It is only
  meaningful on closed paths at $t=T$. Pseudoinverse control is in general not repetitive:
  C. A. Klein, C.-H. Huang, *IEEE Trans. SMC* SMC-13(2):245–250, 1983,
  DOI [10.1109/TSMC.1983.6313123](https://doi.org/10.1109/TSMC.1983.6313123). The plain ZNN
  model therefore shows a nonzero drift. Measured with Euler, τ=1e-3, γ=10:

  | robot | circle | lissajous | plum | heart |
  |---|---|---|---|---|
  | PUMA | 0.007 | 0.000 | 0.012 | 0.004 |
  | Panda | 0.33 | 0.0006 | 0.51 | 0.08 |

  Values are in rad.
* `StateRecorder(disc, every=k)` is a transparent `Discretizer` wrapper that stores
  `x_first`, `x_last` and every k-th state, because `simulate` returns no states. See
  `CORE_CHANGE_REQUESTS.md`, section `problems_robot`.
  `simulate_drift(problem, model, disc, tau, **kw)` returns `(sim, drift)`, with inf for
  diverged trajectories.
* `qlim_violation(theta, qlim)` returns the largest joint-limit violation per trajectory. The
  problem itself does not enforce limits.

## 4. Measured baseline (ZNN, linear activation, Euler, τ = 1e-3, γ = 10, T = 10 s, start on path)

| robot | circle | lissajous | plum | heart |
|---|---|---|---|---|
| PUMA (Corke + tool) | 5.8e-6 | 1.1e-5 | 3.0e-5 | 4.0e-6 |
| Panda (flange) | 6.2e-6 | 1.1e-5 | 3.0e-5 | 4.0e-6 |

Values are the maximum residual in m. $\sigma_{\min}(J)$ along every default path is at least
0.21, and no default path violates a joint limit; both properties were measured, and the
Panda circle case is also tested.

## 5. Options

| option | meaning |
|---|---|
| `robot`, `path` | the arm and the desired path |
| `T` | horizon |
| `path_T` | path period (default T) |
| `theta0` | nominal start configuration |
| `x0_spread` | half-width of the uniform perturbation of θ0 (default 0, start on path) |
| `damping` | λ of the damped pseudoinverse |
| `dh` | PUMA: `corke` or `jin2017`. Panda: `franka`. |
| `tool_d` | tool length beyond the last frame or flange |
| `ee` | Panda: `flange` or `tcp` |
| `base_z` | base height |
| `path_kw` | keyword arguments of the path class, e.g. `radius`, `tilt`, `phase` (not `lissajous`), `A`, `a`, `r`, `n` |
| `center` | `None` (default): anchor the path at $f(\theta_0)$. A 3-vector: absolute curve origin, as in the source papers. |

## 6. Tests (`tests/test_problems_robot.py`; 50 tests, about 17 s, peak RSS 80 MB)

1. **DH tables.** The constants equal the transcribed sources: Corke PUMA with the
   `mdl_puma560.m` limits, the Jin 2017 table (a, d and alpha), and the Franka table with
   flange and FER limits. The nominal poses lie inside the limits.
2. **FK reference values.**
   * PUMA (no tool): `qz` gives (0.4521, −0.15005, 0.4318), `qr` gives
     (0.0203, −0.15005, 0.8636), `qs` gives (0.8636, −0.15005, −0.0203), and `qn` gives
     $((a_2-a_3+d_4)/\sqrt2,-d_3,(a_2-a_3-d_4)/\sqrt2)$. All four are derived by hand from the
     table. The `qz`/`qr` values also equal Corke's printed `fkine` results.
   * `qn` with $\theta_3=+\pi$ (Corke) and with $-\pi$ (here) give the same pose.
   * The pedestal option (`base_z=PUMA560_PEDESTAL`) shifts z by 0.67183 m and leaves J
     unchanged.
   * Panda at q = 0: the flange is at (0.088, 0, 0.926), from the hand calculation
     $(a_7, 0, d_1+d_3+d_5-d_F)$, and the TCP is 0.1034 m lower.
3. **FK against independent implementations.**
   * The FK equals explicit 4×4 products of elementary transforms, at random in-limit
     configurations, for 5 arm variants.
   * For the Panda, the FK also equals the **URDF chain of franka_description**, an
     independent source and parameterisation, for the flange and the TCP.
   * The ready pose gives flange (0.3069, 0, 0.5903) and TCP (0.3069, 0, 0.4869).
4. **Jacobian.**
   * The analytic Jacobian matches central finite differences (h = 1e-6, atol 1e-8) for all
     variants.
   * Without a tool, the PUMA wrist columns vanish.
   * For every variant, the last column is identically zero and the remaining columns have
     rank 3, so the effective redundancy is n − 4.
5. **Problem API.** The registry keys exist and every method returns the right shape.
6. **Each path, on both robots:**
   * the path starts at the end-effector: $E(\theta_0, d(0))=0$;
   * the path is closed: $r_d(T)=r_d(0)$;
   * `dEdt` matches a central finite difference of E in t, and $\dot r_d$ matches one of $r_d$;
   * smooth paths have zero speed at $t=0$ and $t=T$.
7. **`solve_J`.**
   * $J\,dx=r$ with J taken from finite differences;
   * $dx=\mathrm{pinv}(J)r$, the min-norm solution;
   * `JT` equals the dense transpose;
   * the damped formula is correct;
   * on a planar arm, whose $JJ^T$ is exactly singular, `solve_J` falls back to the pinv;
   * damping keeps $\dot\theta$ bounded at the stretched-elbow PUMA singularity.
8. **Batch consistency.** Per-row results equal batched results, and the cache cannot go stale
   after in-place edits.
9. **ZNN-Euler tracking** (8 cases, robot × path, plus the jin2017 variant): the maximum
   residual is below 1e-3 m at τ=1e-3, γ=10, T=10.
10. **Continuous-time property.** With RK4 at τ=1e-3, γ=5, starting off the path, the
    residual equals $\|E(0)\|e^{-\gamma t}$ to a relative error below 1e-5; 1.3e-7 was
    measured. This is the exact error dynamics $\dot E=-\gamma E$ of Jin et al. (2017),
    eq. (12).
11. **Empirical order.** On the PUMA Lissajous, forward-Euler ZNN has an MSSRE slope of
    2 ± 0.15 at fixed $h=\gamma\tau=0.1$ and 1 ± 0.15 at fixed γ=10 (measured 1.99 and 1.00).
12. **Off-path start** (`x0_spread=0.05`): the residual decays by more than 20× in 0.5 s and
    ends below 1e-4. GNN keeps a lag floor that shrinks by more than 3× from γ=100 to γ=1000.
13. **Absolute-centre Jin 2017 circle.**
    * $r_d$ and $\dot r_d$ equal the closed form
      $c+R[\cos\omega t,\sin\omega t\cos30°,\sin\omega t\sin30°]$ with $c=[0.25,0,1.3]$.
    * The start is off the path, with initial error above 0.3 m.
    * ZNN-Euler with γ=5 decays monotonically to below 1e-4 m within 4 s.
14. **Backward-difference information** (`info="bd1"`): tracking stays below 1e-3 m.
15. **Recorder and helpers.** `StateRecorder` reproduces the final residual of `simulate`.
    `joint_drift` and `simulate_drift` are consistent. The default Panda circle stays inside
    the joint limits.

## 7. What was not verified or reproduced

* No ZNN robotics figure was numerically reproduced. The papers start from random or
  unreported configurations and use QP or manipulability terms, so they are not directly
  comparable to the plain tracking problem. The absolute-centre option reproduces the *task*
  of Jin et al. (2017), but not their manipulability-optimising solver.
* The default circle radius and anchoring and the heart scale are assumptions, as flagged
  above.
* Corke's printed `fkine` values for `qz` and `qr` were not seen in a primary printout.
  They are now backed by an independent closed-form standard-DH product and by hand
  derivations, including `qs` and `qn`. The Panda ready-pose TCP is backed by the
  franka_description URDF chain.

## 8. Verification (adversarial review)

An independent verifier re-derived every claim in this area from primary sources and
independent numerical checks, using scratch scripts outside the repository.

**Sources re-fetched and compared symbol by symbol**

| item | source | verdict |
|---|---|---|
| Corke PUMA 560 DH table (a, d, alpha), named poses qz/qr/qs/qn, `qlim` | `robotics-toolbox-matlab/models/mdl_puma560.m` (raw GitHub) | correct |
| Python PUMA (pedestal 26.45 in, qn, active qlim) | `robotics-toolbox-python/src/roboticstoolbox/models/DH/Puma560.py` | Link parameters and pedestal are correct. **Two attribution errors were fixed.** The Python qn has $+\pi$, not $-\pi$. The implemented limits are the MATLAB ones, not the Python active ones. |
| Jin, Li, La, Luo 2017, Table I | authors' PDF, `ara.cse.unr.edu/wp-content/uploads/2014/12/IEEE-TIE_Jin-Li-La-Luo2017.pdf` | correct: a = [0, .4318, .4318, 0, 0, 0], alpha = [π/2, 0, −π/2, π/2, −π/2, 0], d = [.67, 0, .15005, 0, 0, .2] |
| Jin 2017, eqs. (9), (12), (13) | same PDF | correct: $\mu=\sqrt{\det JJ^T}$ and $Jw=-k_0(f(\theta)-r_d)+\dot r_d$. Our design formula with linear Φ and γ = k0 is exactly eq. (12). |
| Jin 2017, Sec. VI-B circle | same PDF | Radius 0.2, 30° about X and 0.2 rad/s are correct. The spec **omitted** the centre [0.25, 0, 1.3], k0 = 5 and Ω = [−0.5, 0.5]^6; these are now documented, and the `center` option was added. |
| Franka DH table (Craig) and flange 0.107 | `frankarobotics.github.io/docs/robot_specifications.html` | correct; the page labels the table FR3 |
| FER joint limits | same page, "Franka Emika Robot (FER / Panda)" table | correct |
| Panda kinematics, independent | `franka_ros/franka_description/robots/common/franka_arm.xacro` | identical to 1e-12 (tested) |
| Corke Panda.py (flange 107 mm, tool 103 mm, limits) | `robotics-toolbox-python/.../DH/Panda.py` | consistent |
| Yang et al. 2025, eqs. (29)–(31), Lissajous A = B = 0.06, a = 3, b = 2, δ = π/2, plum r = 0.1, n = 5, phase $2\pi\sin^2(\pi t/2T)$ | arXiv:2503.01857 (HTML full text) | Correct; heart scale a, z0 and T are not given (the assumption stands). The QP description was **corrected**: ρ = ι(q − q0) with ι = 1, there is no position feedback, and q̇ bounds are ±0.65 rad/s. |
| DOIs (Corke & Armstrong 1994, Haddadin 2022, Yang 2025, Corke & Haviland 2021, Klein & Huang 1983, Zhang & Zhang 2013, Li-Zhang-Jin 2017, Jin 2017, Corke 2017 book) | Crossref API | all resolve to the cited titles, venues and pages |

**Independent numerical checks**

* An independent closed-form standard-DH FK reproduces qz, qr, qs and qn exactly, and the
  jin2017 nominal pose gives p0 = (0.6107, −0.15005, 0.87).
* The modified-DH Panda matches the URDF chain to 2e-16. The ready-pose flange is at
  (0.30689, 0, 0.59028) and the TCP at (0.30689, 0, 0.48688).
* **New finding:** the last column of the position Jacobian is zero (below 1.1e-16) on every
  arm variant, so the effective redundancy is n − 4 (Sec. 0).
* All claims in Secs. 3–4 were re-measured: the max residuals (PUMA 5.8e-6, 1.1e-5, 3.0e-5,
  4.0e-6; Panda 6.2e-6, 1.1e-5, 3.0e-5, 4.0e-6), the joint drifts, zero joint-limit violations
  on the default paths, $\sigma_{\min}\ge0.217$ along the paths, and the radius-0.2 violations
  (PUMA joint 3 by 0.26 rad, Panda joint 2 by 0.49 rad).
* RK4 reproduces $\|E(t)\|=\|E(0)\|e^{-\gamma t}$ to a relative error of 1.3e-7. The
  forward-Euler orders are 1.99 at fixed h and 1.00 at fixed γ.
* Path derivatives were checked by hand: smooth phase
  $\dot\phi=(\pi^2/T)\sin(\pi t/T)$, the plum, heart and Lissajous derivatives, and the
  Craig-convention link transform $R_x(\alpha)T_x(a)R_z(\theta)T_z(d)$ with translation
  $(a,-d\sin\alpha,d\cos\alpha)$. All are correct.

**API contract (DESIGN.md)**

* Batch shapes are (B, n), (B, 3) and (B, 3, n).
* `data` contains `'t'`.
* `dEdt` is exact.
* `solve_J` returns the min-norm pseudoinverse solution.
* The registry keys `robot_puma560` and `robot_panda` are unique.
* The robot area made no edits to core files; `CORE_CHANGE_REQUESTS.md` has only an
  appended section.
* The tests are fast: 17 s and 80 MB.
