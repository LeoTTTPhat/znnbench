# Spec: discretization zoo (`zead.py`, `uhlig.py`, `adams.py`)

Area owner files: `znnbench/discretizations/zead.py`, `znnbench/discretizations/uhlig.py`,
`znnbench/discretizations/adams.py`, `tests/test_discretizations.py`, this file.
Reference classes reused (not edited): `LookAhead`, `euler`, `taylor_zhang` in `multistep.py`.

## 1. Conventions

**Look-ahead (ZeaD-type) formula.** Every formula with $N$ instants is stored in *tau-form*

$$
\dot x_k=\frac1\tau\sum_{i=0}^{N-1}c_i\,x_{k+1-i}+O(\tau^{p}),\qquad
\sum_i c_i=0,\quad \sum_i c_i(1-i)=1,\quad \sum_i c_i(1-i)^m=0\ (2\le m\le p).
$$

Coefficients are transcribed as exact fractions whenever the source prints rationals.
The causal update (`LookAhead.step`) is $x_{k+1}=(\tau\dot x_k-\sum_{i\ge1}c_ix_{k+1-i})/c_0$ with
$\dot x_k=f(t_k,x_k)$ from the continuous model; the first $N-2$ steps use forward Euler (the start-up used by
Guo et al. 2017, Uhlig 2024, Yang et al. 2020). The auxiliary state $z$ is advanced by forward Euler.

**Residual order.** $x^*(t)$ satisfies the ZNN vector field exactly, so the local error is
$O(\tau^{p+1})$ and, at fixed $h=\gamma\tau$, the steady-state residual of linear ZNN is
$O(\tau^{p+1})$: `expected_order = p + 1`.

**Linear-ZNN stability at fixed h** (Zhang, Gong, Yang, Li, Yang, IEEE TNNLS 30(3) 2019, eq. (17)):

$$
c_0e_{k+1}+(c_1+h)e_k+c_2e_{k-1}+\dots+c_{N-1}e_{k+2-N}=O(\tau^{p+1}),\qquad
P_h(z)=c_0z^{N-1}+(c_1+h)z^{N-2}+\sum_{i\ge2}c_iz^{N-1-i}.
$$

`znn_stability` returns the effective interval $(0,h_{\max})$ (all roots of $P_h$ in the open unit disk; first
crossing refined by bisection) and $h_{\rm opt}=\arg\min_h\max|{\rm roots}\,P_h|$ (the "optimal stepsize"
criterion of Zhang et al. 2019). **All $h$ in ZNN-Bench are $h=\gamma\tau$ with tau-form coefficients**; papers
that scale differently are converted below.

**Zero-stability vs. ZNN-stability.** `root_condition` checks the Dahlquist root condition of
$\rho(z)=\sum_i c_iz^{N-1-i}$ (roots in the closed disk, unit roots simple) and the *strong* root condition
($z=1$ is the only unit root). Zero-stability alone is not sufficient for ZNN: the symmetric Euler formula is
zero-stable but $P_h$ has a root $<-1$ for every $h>0$ (Sun, Tian, Wang 2019, Thm 3.3; Uhlig's "mystery",
JDEA 2019 item (A)). For small $h>0$ the principal root moves to $1-h/\rho'(1)=1-h$, so the strong root
condition is what guarantees a non-empty interval.

## 2. ZeaD formulas (`zead.py`)

| key | $N$ | tau-form coefficients (integer / denominator) | $p$ | source (DOI) |
|---|---|---|---|---|
| `taylor_zhang` *(multistep.py, verified)* | 4 | (2, −3, 2, −1)/2 | 2 | Zhang, Jin, Guo, Yin, Chou, JCAM 273 (2015) 29–40, doi:10.1016/j.cam.2014.05.027; range: Zhang et al. TNNLS 2019, doi:10.1109/TNNLS.2018.2861404 |
| `zead_4i_general(a1)` | 4 | $(\tfrac12-a_1,\ 3a_1,\ -3a_1-\tfrac12,\ a_1)$, $a_1<0$; default $a_1=-\tfrac14$ → (3,−3,1,−1)/4 | 2 | Hu, Kang, Zhang, Neurocomputing 306 (2018) 108–118, doi:10.1016/j.neucom.2018.03.053 (formula as reproduced in Sun, Tian, Wang, DDNS 2019, Table 1, doi:10.1155/2019/4745759) |
| `zead_4i_6321` | 4 | (6, −3, −2, −1)/10 ($a_1=-\tfrac1{10}$) | 2 | Guo, Lin, Su, Sun, Huang, Numer. Algor. 77 (2018) 23–36, doi:10.1007/s11075-017-0302-4; range (0, 0.8): Zhang, Qi, Li, Qiu, Yang, Numer. Algor. 81 (2019) 561–574, doi:10.1007/s11075-018-0561-8 |
| `zead_4i_fifd` | 4 | (5, −3, −1, −1)/8 ($a_1=-\tfrac18$) | 2 | Li, Mao, Uhlig, Zhang, JCAM 327 (2018) 155–166, doi:10.1016/j.cam.2017.06.017 (eqs. (14),(21), via Uhlig 2019 item (D)) |
| `zead_5i` | 5 | (8, 1, −6, −5, 2)/18 | 3 | Li, Mao, Uhlig, Zhang, Numer. Algor. 81 (2019) 609–629, doi:10.1007/s11075-018-0564-5 (eqs. (23),(27); via Uhlig 2019 (E) and Uhlig 2024 eq. (4i)) |
| `zead_5i_general(a1)` | 5 | $(a_1+\tfrac13,\ \tfrac12-4a_1,\ 6a_1-1,\ \tfrac16-4a_1,\ a_1)$, $a_1\in(\tfrac1{12},\tfrac16)$; default $\tfrac19$ = `zead_5i` | 3 | Zhang, He, Hu, Guo, Li, Shi, JCAM 347 (2019) 314–329, doi:10.1016/j.cam.2018.08.017 (via Sun–Tian–Wang Table 1) |
| `zead_6i_guo2017` | 6 | (24, −5, −12, −6, −4, 3)/48 | 3 | Guo, Nie, Yan, IEEE TSMC-S 47(8) (2017) 2301–2310, Thm 4 eq. (6), doi:10.1109/TSMC.2017.2656941 |
| `zead_6i_qiu2019` | 6 | (13, −6, −2, −4, −3, 2)/24 ("6NτCD" in Uhlig's list) | 3 | Qiu, Zhang, Guo, Yang, Li, Numer. Algor. 81 (2019) 1043–1065, eqs. (16),(18), doi:10.1007/s11075-018-0581-4 (via Uhlig 2019 item (F)) |
| `zead_7i_general(p1,p2)` | 7 | eq. (8) below; default $(p_1,p_2)=(-1,-1)$ → (300, 161, −300, −300, 100, 75, −36)/780 | 4 | Yang, Zhang, Hu, Qiu, IEEE TNNLS 31(9) (2020) 3204–3214, doi:10.1109/TNNLS.2019.2938866 |
| `zead_8i_wu2020` | 8 | (1000, −51, −400, −600, −200, 175, 176, −100)/2220 | 4 | Wu, Zhang, Guo, Li, Ming, IEEE Access 8 (2020) 228188–228198, eq. (12), doi:10.1109/ACCESS.2020.3045780 |
| `zead_8i_chen2019` | 8 | $\tfrac{100}{211},-\tfrac{1613}{12660},0,-\tfrac{90}{211},0,\tfrac{15}{844},\tfrac{124}{1055},-\tfrac{35}{633}$ | 4 | Chen, Zhang, IEEE Access 7 (2019) 125909–125918, doi:10.1109/ACCESS.2019.2938840, with $\eta_1=\eta_3=0,\eta_2=-\tfrac9{10}$ as given in Chen, Kang, Zhang, Mathematics 11 (2023) 3354 eq. (7), doi:10.3390/math11153354 |
| `zead_10i_wu2020(row)` | 10 | $d_0..d_9$ of Table 1, rows 1–6 (row 1 default) | 5 | Wu et al. 2020, eq. (14), Table 1 |
| `zead_11i_chen2023` | 11 | $\tfrac{42}{101},\tfrac{90583}{1272600},-\tfrac{126}{505},-\tfrac{126}{505},-\tfrac{252}{2525},0,\tfrac{378}{2525},\tfrac{28}{505},-\tfrac{414}{3535},\tfrac{231}{20200},\tfrac{56}{4545}$ | 5 | Chen, Kang, Zhang, Mathematics 11 (2023) 3354, Thm 3 eq. (6) |

General 7-instant formula (Yang et al. 2020, eq. (8)):

$$
\dot x_k\doteq\frac{-5x_{k+1}+\tfrac{214p_1+77p_2+130}{60}x_k-5p_1x_{k-1}-5p_2x_{k-2}
+\tfrac{5(2p_1+4p_2+5)}{3}x_{k-3}-\tfrac{5(2p_1+3p_2+6)}{4}x_{k-4}+\tfrac{3p_1+4p_2+10}{5}x_{k-5}}{(2p_1+p_2-10)\tau}+O(\tau^4),
$$

effective domain (eq. (16)): $p_1-7p_2-5>0$, $204p_1^2+196p_2^2+419p_1p_2+800p_1+525p_2+500<0$, $p_1<0$
(exact extent of the region: $p_1\in(-1.4,-0.6)$, $p_2\in(-1.0267,-0.8)$; corrected by the verifier from
$(-1.38,-0.62)$, $(-1.02,-0.82)$).

Ten-instant Table 1 (Wu et al. 2020), $d_0..d_9$:

| row | coefficients |
|---|---|
| 1 | 175/432, 737/7560, −35/144, −245/864, −35/216, 35/288, 343/2160, −35/864, −95/1008, 35/864 |
| 2 | 350/823, 9839/691320, −105/823, −280/823, −105/823, 35/823, 777/4115, 70/2469, −935/5761, 385/6584 |
| 3 | 700/1677, 21587/469560, −280/1677, −560/1677, −70/559, 140/1677, 1148/8385, 70/1677, −1810/11739, 245/4472 |
| 4 | 100/237, 947/66360, −20/237, −100/237, −10/79, 40/237, 88/1185, 10/237, −230/1659, 95/1896 |
| 5 | 350/873, 31063/244440, −280/873, −175/873, −140/873, 35/873, 952/4365, −35/873, −670/6111, 35/776 |
| 6 | 175/428, 7051/89880, −175/856, −35/107, −105/856, 35/428, 819/4280, −35/642, −555/5992, 35/856 |

(Table 1 is an image in the source; transcription verified exactly: each row satisfies the order-5 moment
conditions in rational arithmetic, which a single misread digit would break.)

**Step-size statements in the literature (converted to $h=\gamma\tau$):**
Euler $(0,2)$, fastest at $h=1$ (Jin & Zhang, IEEE TNNLS 26(7) 2015 1525–1531, doi:10.1109/TNNLS.2014.2342260,
Thm 1/Remark 3); Taylor–Zhang $(0,1)$, $h_o=0.2400$, $M(h_o)=0.7938$ (Zhang et al. TNNLS 2019, Thm 1, Table I);
ZeaD 6321 $(0,0.8)$ (Zhang, Qi et al. 2019, abstract). Guo et al. 2017 define $h=2\mu\tau$ (their $h$ = 2× ours)
and state no interval. No other source stated an interval that we could access.

## 3. Uhlig's convergent look-ahead formulas (`uhlig.py`)

Sources: [U19] F. Uhlig, J. Difference Equ. Appl. 25 (2019) 930–941, doi:10.1080/10236198.2019.1627343
(arXiv:1904.10569); [UF] F. Uhlig, Linear Multilinear Algebra 68 (2020) 1894–1910,
doi:10.1080/03081087.2019.1648375 (arXiv:1904.10568); [U24] F. Uhlig, Numer. Math. 156 (2024) 691–739,
doi:10.1007/s00211-023-01393-5 (arXiv:2008.02724v6).

Construction ([U19] §2): rational Taylor matrix $A_{k+s,k}$, $a_{1,v}=1/(v+1)!$,
$a_{u,v}=(-1)^{v+1}(u-1)^{v+1}/(v+1)!$; $R={\rm rref}(A^T)=[I_k,B]$; seed $y\in\mathbb R^s$ gives
$q=[-By;y]/q_1$, $p=[1,-\sum q,q_2,\dots]$, $c=\sum_ip_i(1-i)$, rule
$x_{j+1}=-(p_2x_j+\dots)+c\,\tau\dot x_j$; ZNN truncation order $k+2$. Implemented exactly in
`uhlig_seed_polynomial`; it reproduces both runs printed in [U19] p.7 bit-for-bit.

| key (j_s of [U24]) | label in [U19]/[UF] | polynomial | ZNN order | notes |
|---|---|---|---|---|
| `uhlig_2_3` | 2_2 / 2_2b | $8x^4+x^3-6x^2-5x+2$, $c=9/4$; seed $y=[-5,2]$ | 4 | identical to `zead_5i` |
| `uhlig_3_4(a)` | 3_3 | seed $y=[a,110,-40]$, $1\le a\le2.5$; $a=1$: $[237,80,-182,-206,1,110,-40]/237$, $c=196/79$ | 5 | tau-form (237, 80, −182, −206, 1, 110, −40)/588 |
| `uhlig_4_6` | 4_5a | $p_9$ printed to 16 digits in [UF] p.6 (non-rational) | 6 | 10 instants; moments hold to $10^{-15}$ |

**Naming (ASSUMPTION / finding).** [U19]/[UF] name a polynomial "k_s" with degree $k+s$ ($k+s+1$ instants);
[U24] uses "j_s" with $j+s$ instants and local order $j+2$ (the same 5-IFD is "2_2" in [U19]/[UF] and "2_3" in
[U24]; [U24] also calls a 9-IFD "4_5"). Keys follow [U24]; [UF]'s "4_5a" (10 instants, order 6) therefore
becomes `uhlig_4_6`. The original label is kept in `describe()['label']`.

**Not reproduced: orders 7 and 8.** [U19]/[U24] report convergent formulas of ZNN truncation orders 7 and 8
("types j_s with 1 ≤ j ≤ 6"), and [UF] tests "5_5" and "5_6b" (order 7), but their coefficients are only in
Uhlig's MATLAB files (`runconv1step.m`, `Polyksrestcoeff3.m`, `auburn.edu/~uhligfd/m_files/...`), which
returned 404 (no Wayback copy reachable). We did **not** regenerate them with our own seed search, since that
would produce formulas that are not Uhlig's published ones.

## 4. Adams–Bashforth (`adams.py`)

$x_{k+1}=x_k+\tau\sum_{j=0}^{q-1}\beta_jf(t_{k-j},x_{k-j})$ with stored past evaluations (causal):
AB2 $\beta=(3,-1)/2$, AB3 $(23,-16,5)/12$, AB4 $(55,-59,37,-9)/24$ (Hairer, Nørsett, Wanner, *Solving ODEs I*,
§III.1). ZNN usage: Qiu et al. Numer. Algor. 81 (2019) compare with an AB-type DTZD ("DTZD-AB"; AB order
not verified from the full text), doi:10.1007/s11075-018-0581-4.
Error polynomial at fixed $h$: $P_h(z)=z^q-z^{q-1}+h\sum_j\beta_jz^{q-1-j}$; intervals $(0,1)$, $(0,6/11)$,
$(0,3/10)$; `expected_order = q+1` (local error $C_{q+1}\tau^{q+1}x^{*(q+1)}$, independent of $\gamma$).
ASSUMPTIONS: lower-order AB start-up; $z$ by forward Euler unless `z_scheme="ab"`.

## 5. Computed properties and empirical orders

Linear ZNN on `linear_system(n=4, kappa=10, seed=1)`, $x_0=x^*(0)$, $h=h_{\rm opt}$, MSSRE over the last 25 %
of $T$, least-squares slope with round-off floor $10^{-11}$ (all values from the test configuration):

| key | instants | expected | 2nd root mod. ρ(z) | $h_{\max}$ | $h_{\rm opt}$ | ρ at $h_{\rm opt}$ | τ range | slope |
|---|---|---|---|---|---|---|---|---|
| `euler` | 2 | 2 | 0 | 2.0000 | 1.0000 | 0.0000 | 0.04–0.005 (calibration run; tested in test_core) | 2.00 |
| `taylor_zhang` | 4 | 3 | 0.7071 | 1.0000 | 0.2401 | 0.7937 | 0.04–0.005 (calibration run; tested in test_core) | 2.97 |
| `zead_4i_general` | 4 | 3 | 0.5774 | 2.0000 | 0.3894 | 0.6934 | 0.04–0.005 | 2.99 |
| `zead_4i_6321` | 4 | 3 | 0.4082 | 0.8000 | 0.6000 | 0.5774 | 0.04–0.005 | 2.99 |
| `zead_4i_fifd` | 4 | 3 | 0.4472 | 1.0000 | 0.5887 | 0.5848 | 0.04–0.005 | 2.99 |
| `ab2` | 2-step | 3 | 0 | 1.0000 | 0.6667 | 0.5774 | 0.04–0.005 | 2.99 |
| `zead_5i` | 5 | 4 | 0.9025 | 0.2808 | 0.0742 | 0.9284 | 0.02–0.0025 | 3.92 |
| `zead_5i_general`(a1=1/10) | 5 | 4 | 0.8763 | 0.2667 | 0.0956 | 0.9088 | 0.02–0.0025 | 3.94 |
| `zead_6i_guo2017` | 6 | 4 | 0.7627 | 0.3333 | 0.1580 | 0.8536 | 0.02–0.0025 | 3.98 |
| `zead_6i_qiu2019` | 6 | 4 | 0.7604 | 0.6667 | 0.2360 | 0.7886 | 0.02–0.0025 | 3.99 |
| `uhlig_2_3` | 5 | 4 | 0.9025 | 0.2808 | 0.0742 | 0.9284 | 0.02–0.0025 | 3.92 |
| `ab3` | 3-step | 4 | 0 | 0.5455 | 0.3587 | 0.6916 | 0.02–0.0025 | 3.99 |
| `zead_7i_general` | 7 | 5 | 0.9764 | 0.1641 | 0.1388 | 0.9012 | 0.012–0.003 | 4.98 |
| `zead_8i_wu2020` | 8 | 5 | 0.8532 | 0.5189 | 0.1937 | 0.8309 | 0.012–0.003 | 4.99 |
| `zead_8i_chen2019` | 8 | 5 | 0.9159 | 0.4281 | 0.2764 | 0.8671 | 0.012–0.003 | 4.99 |
| `ab4` | 4-step | 5 | 0 | 0.3000 | 0.2146 | 0.8070 | 0.02–0.0025 | 4.98 |
| `uhlig_3_4` | 7 | 5 | 0.9591 | 0.0544 | 0.0255 | 0.9749 | 0.008–0.002 (T=10) | 4.87 |
| `zead_10i_wu2020` | 10 | 6 | 0.9427 | 0.1291 | 0.0603 | 0.9415 | 0.012–0.003 | 5.90 |
| `zead_11i_chen2023` | 11 | 6 | 0.9252 | 0.2229 | 0.0675 | 0.9347 | 0.012–0.003 | 5.91 |
| `uhlig_4_6` | 10 | 6 | 0.9056 | 0.1760 | 0.0930 | 0.9112 | 0.012–0.003 | 5.95 |

Ten-instant rows 2–6: $(h_{\max},h_{\rm opt})$ = (0.1254, 0.0417), (0.1838, 0.0379), (0.1948, 0.0676),
(0.0566, 0.0313), (0.3610, 0.0667); empirical slopes 5.73–5.97 (exploration runs, T = 6–12).
`uhlig_3_4(a)`: $h_{\max}$ = 0.0544, 0.0825, 0.1111, 0.1404 for a = 1, 1.5, 2, 2.5.

**Every claimed order was reproduced** (all |slope − expected| ≤ 0.14 in the test configuration).

## 6. Findings, caveats and flagged assumptions

1. *Stability shrinks with order.* $h_{\max}$ falls from 2 (Euler) to 0.05–0.36 for order-5/6 formulas and the
   second root of $\rho$ approaches the unit circle (0.90–0.98), so the transient decays like
   $\rho_{\rm opt}^k$ with $\rho_{\rm opt}\approx0.93$–$0.99$: thousands of steps before the steady state.
   Example: `uhlig_3_4(a=2.5)` ($\rho_{\rm opt}=0.988$) gives erratic slopes (6.5–12) at T = 6–12 s but 4.99 at
   T = 40 s.
2. *Frozen-coefficient intervals need $\tau\|A^{-1}\dot A\|\ll h$.* The Jacobian of the linear-ZNN field is
   $-\gamma I-A^{-1}\dot A$, so the per-step perturbation of $P_h$ is $\tau\|A^{-1}\dot A\|$ (2.6–4.1 here), which at
   τ ≈ 0.03 is as large as $h_{\rm opt}$ of the order-5/6 formulas. Observed at $h=h_{\rm opt}$ (T = 20 s):
   `zead_7i_general` diverges for τ ≥ 0.035 (residual 4e6 at τ = 0.025), `uhlig_4_6` diverges for τ ≥ 0.05
   (2e4 at 0.035); `zead_5i`, `zead_8i_chen2019`, `uhlig_3_4`, `ab4` have residuals 4e1–1e4 at τ = 0.1 (up to
   3e10 for `ab4` at τ = 0.2) although $h\in(0,h_{\max})$. Low-order formulas (Euler, 4-instant, AB2) stay accurate.
3. *Misprint in Guo, Nie, Yan (2017), Thm 5:* the printed characteristic polynomial
   $\vartheta^5+\tfrac5{24}\vartheta^4+\tfrac12\vartheta^3+\tfrac14\vartheta^2+\tfrac16\vartheta-\tfrac18$ has sign errors
   (no root at 1); the listed roots (0.3596, −0.5211±0.4694i, 0.2372±0.8065i) are those of the misprint. The
   correct polynomial $24z^5-5z^4-12z^3-6z^2-4z+3$ has roots 1, −0.7627, −0.2062±0.6206i, 0.3833 and is
   zero-stable, so the paper's conclusion stands.
4. *Attribution:* Wu et al. (2020) cite Guo, Nie & Yan (2017) for their 8-instant formula (12); the
   sections of Guo et al. we read (I–IV) contain only the 6-instant rule. Flagged, coefficients used as printed
   by Wu et al.
5. *Precision barrier (Yang, Zhang, Hu, Numer. Algor. 88 (2021) 883–902, doi:10.1007/s11075-020-01061-x).*
   Only the abstract was accessible: N = 2,3 → O(τ); 4 → O(τ²); 5,6 → O(τ³); 7,8 → O(τ⁴); 9 → O(τ⁵). Our
   registered formulas attain these maxima for N = 4–8, and numerical scans (test) find no strongly-stable
   member of the N=6/p=4, N=7/p=5 (1-parameter) and N=8/p=5, N=9/p=6 (2-parameter grid) families, while the
   N=5/p=3 and N=7/p=4 families contain stable members — consistent with the barrier (evidence, not proof).
   The 9-instant general formula (Franklin Inst. 2022) was not accessible and is not registered.
6. ASSUMPTION: `zead_7i_general` default $(p_1,p_2)=(-1,-1)$ (a point of domain (16)); the instances in Table II
   of Yang et al. (2020) could not be loaded (image server unavailable).
7. ASSUMPTION: defaults $a_1=-1/4$ (`zead_4i_general`) and $a_1=1/9$ (`zead_5i_general`).
8. Secondary transcriptions: `zead_4i_fifd`, `zead_5i`, `zead_6i_qiu2019` from Uhlig [U19] items (D), (E), (F)
   (primary texts paywalled); general 4-/5-instant families from Sun–Tian–Wang (2019) Table 1;
   `zead_8i_chen2019` from Chen–Kang–Zhang (2023) eq. (7). The 5-instant source's own step-size statement
   (if any) was not accessible; values in §5 are computed.

## 7. Tests (`tests/test_discretizations.py`, ~10 s)

* registry/metadata for all 18 keys (causal, `expected_order`, $0<h_{\rm opt}<h_{\max}$, ρ_opt < 1, DOI);
* exact consistency conditions (rational moments $\mu_0=0,\mu_1=1,\mu_2..\mu_p=0,\mu_{p+1}\ne0$; float tolerance
  for `uhlig_4_6`) and the printed integer vectors; family members reproduce named formulas; the 11-instant
  normalised polynomial equals Appendix D's $P_{10}$; the 7-instant formula equals the characteristic equation in
  the proof of Theorem 2;
* Uhlig's seed construction reproduces both printed TPOLY runs, the printed second-root moduli and the root lists
  of [UF]; the seed family is convergent exactly for the published range;
* zero-stability + strong root condition for every key; published root lists (11-instant; Guo misprint);
  negative controls (symmetric Euler: zero-stable, $h_{\max}=0$; Sun–Tian–Wang order-5 six-instant formula:
  root −6.9614); precision-barrier scans; effective parameter domains of the 4-, 5-, 7-instant families;
* step-size intervals vs. literature (Euler, Taylor–Zhang incl. $h_o=0.24$, $M=0.7938$, ZeaD 6321, AB2–AB4) and a
  simulation check that $0.9h_{\max}$ converges while $1.25h_{\max}$ diverges;
* empirical order at fixed $h=h_{\rm opt}$ on `linear_system` for every key (|slope − expected| < 0.35, ≥ 4
  points above $10^{-11}$), plus ten-instant row 4;
* causality (exactly one evaluation per step, at $t_k$ with integer $k$) and AB order conditions / `z_scheme`.
* *verifier additions:* derivative-approximation order on samples of $e^t$ (independent of the package's moment
  routine); $h_{\max}$ of every key against an independent boundary-locus computation; the Sun–Tian–Wang closed-form
  step range of the general 3-instant formula; domain (16) = strong-root region of formula (8) on a 41×41 grid;
  JSON-serialisable `describe()` and decimal-exact float parameters; batch-row independence and `reset` between runs.

## 8. Verification (adversarial review, 2026-10-18)

An independent verifier re-checked every registered key against the sources they could open and with their own
numerical code (scalar linear-ZNN simulator, boundary-locus stability, brute-force spectral-radius scans), separately
from the implementer's routines.

**Primary/secondary full texts consulted by the verifier.**
[U19] arXiv:1904.10569v1; [UF] arXiv:1904.10568; [U24] arXiv:2008.02724v6; Sun, Tian, Wang, "Multi-step
discrete-time Zhang neural networks with application to time-varying nonlinear optimization", optimization-online
preprint 6960 (2018), i.e. the preprint of DDNS 2019, doi:10.1155/2019/4745759. Crossref metadata confirmed the title,
authors, volume and pages of every DOI cited in this file. The IEEE Xplore, Springer, Elsevier, MDPI and Hindawi pages
could not be opened by the verifier (paywall or bot protection; web-search budget exhausted), so items that depend
only on those texts are marked *unverifiable* below.

| item | verdict | evidence |
|---|---|---|
| `zead_4i_fifd` (5,−3,−1,−1)/8 | correct | [U19] item (D), symbol by symbol (eqs. (14),(21) of Li–Mao–Uhlig–Zhang JCAM 2018) |
| `zead_5i`, `uhlig_2_3` (8,1,−6,−5,2)/18 | correct | [U19] item (E); [U24] eq. (4i) "type j_s = 2_3"; [UF] p4 and its roots |
| `zead_6i_qiu2019` (13,−6,−2,−4,−3,2)/24 | correct | [U19] item (F); label is "6NτCD" there (implementer's "6NgCD" renamed) |
| `zead_4i_6321` (6,−3,−2,−1)/10 | correct | Sun–Tian–Wang Table 1, ref. [12] = Guo, Lin, Su, Sun, Huang, Numer. Algor. 77 (2018); also [U19] item (C). $h_{\max}=0.8$ exactly in our scale (range (0,0.8) of Zhang–Qi et al. 2019: abstract not opened, consistent) |
| `zead_4i_general` family, domain $a_1<0$ | correct | Sun–Tian–Wang Table 1 ref. [16] (= Hu, Kang, Zhang 2018); domain equals the strong-root region on a 7001-point grid; default $a_1=-1/4$ attribution (Chen–Kang–Zhang eq. (9)) unverifiable |
| `zead_5i_general` family, domain $(1/12,1/6)$ | correct | Sun–Tian–Wang Table 1 ref. [17] (= Zhang, He, Hu, Guo, Li, Shi, JCAM 2019); domain equals strong-root region on a 10001-point grid |
| Sun–Tian–Wang six-instant $O(\tau^5)$ negative control, root −6.9614 | correct | preprint eq. (24) and the printed $\rho(\gamma)$ and roots |
| symmetric Euler (negative control) | correct | Sun–Tian–Wang Thm 3.1/3.3; [U19] item (A) |
| `uhlig_3_4(a)` seed $[a,110,-40]$, $1\le a\le2.5$ | correct | [U19] p.7 TPOLY printed exactly (incl. 446/465, 196/79); the family is strongly stable for $a\in(0,\approx2.65]$, which contains Uhlig's range |
| `uhlig_4_6` ($p_9$ of 4_5a) | correct | all ten 16-digit coefficients compared with [UF] p.6; roots match [UF] |
| Uhlig construction (`uhlig_seed_polynomial`) | correct | [U19] §2 ($q=[-R[0;y];y]/q_1$, $p=[1;-\sum q;q_{2:}]$), reproduces both printed runs |
| j_s relabelling (2_2→2_3, 3_3→3_4, 4_5a→4_6) | correct | [U24]: "5-IFD … of type j_s = 2_3", "9-IFD formula of type 4_5 with local truncation order 6", Euler "type 1_2": j+s = number of instants, local order j+2 |
| Uhlig's $h=\eta\tau$ vs. ours | correct | tau-form division by $c$ leaves $h$ unchanged; [UF] optimal $h$ for 4_5a (0.03–0.057) and for 3_3 (0.027–0.032) lie inside our $(0,h_{\max})$ |
| `zead_7i_general` formula (8) and domain (16) | correct (indirect) | full text unverifiable; but the published domain (16) coincides exactly with the computed strong-root region of the transcribed formula (8) on a 241×241 grid (0 mismatches), and the moments hold exactly; a misprint in either would break this |
| `zead_6i_guo2017`, Guo misprint finding, "$h=2\mu\tau$" | unverifiable | IEEE full text not accessible; coefficients satisfy the order-3 conditions exactly |
| `zead_8i_wu2020`, `zead_10i_wu2020` rows 1–6 | unverifiable (strong indirect evidence) | IEEE Access page needs JavaScript; every row satisfies the order-4/5 moment conditions exactly in rational arithmetic (3- and 4-parameter families, so a misread digit would break them) |
| `zead_8i_chen2019`, `zead_11i_chen2023` | unverifiable (strong indirect evidence) | MDPI/IEEE blocked; exact moments; normalised coefficients consistent with $\eta_1=\eta_3=0,\eta_2=-9/10$; 11-instant coefficients consistent with the separately printed Appendix D roots (as transcribed by the implementer) |
| Taylor–Zhang range (0,1), $h_o=0.24$, $M=0.7938$ | consistent | computed 1.0000 / 0.2401 / 0.7937 (paper not opened; the 4-digit agreement of $M$ indicates the same $h$ convention) |
| Euler (0,2), $h=1$ | correct | standard |
| AB2–AB4 coefficients, intervals (0,1), (0,6/11), (0,3/10) | correct | textbook; closed form at $z=-1$ and boundary locus |
| AB "DTZD-AB (AB4)" usage, TII-2021 citation | unverifiable / removed | AB order in Qiu et al. not checked; the TII 2021 citation could not be traced to any consulted source and was removed |

**Independent numerical checks (verifier's own code).**
* Moments: recomputed directly; all 15 look-ahead keys have exactly the claimed order $p$ ($\mu_{p+1}\neq0$).
* $h_{\max}$: boundary-locus method agrees with `znn_stability` to $<10^{-5}$ for all 18 keys; $h_{\rm opt}$ and
  $\rho_{\rm opt}$ agree with a 20001-point brute-force scan to $10^{-4}$.
* Empirical order with a separate scalar simulator ($a(t)x=b(t)$, exact start values, T = 20 s, $h=h_{\rm opt}$):
  slopes 2.99–3.00 (4-instant, AB2), 3.83–3.99 (5/6-instant, AB3; the 5-instant slope is still pre-asymptotic,
  successive error ratios 13→14.5→15.7 towards 16), 4.97–4.99 (7/8-instant, AB4 4.96), 4.64 (`uhlig_3_4`, slow decay
  $\rho_{\rm opt}=0.975$), 5.82–5.91 (10/11-instant, `uhlig_4_6`). This agrees with the package's own tests.
* The breakdown examples of §6.2 were reproduced (`zead_7i_general` 3.8e6 at τ = 0.025 and divergence at 0.035;
  `uhlig_4_6` 1.9e4 at 0.035 and divergence at 0.05; `ab4` 1.1e4 at τ = 0.1).
* API contract: one f evaluation per step at $(t_k,x_k)$ with integer k (causal); batch rows with different γ equal
  separate runs; `simulate` resets the history; no core file edited.

**Bugs found and fixed by the verifier.**
1. `integer_form` used `np.gcd`. It overflowed int64 for large denominators and returned garbage such as
   `[0, 0, 0, 0] / 0`. It also put numpy ints into `describe()`, so `json.dumps(describe())` failed for all 14
   exact-coefficient keys. It now uses Python `math.gcd`.
2. Float family parameters (`a1=-0.1`, `p1=-1.1`, `a=1.5`) were converted to their binary expansions, e.g.
   `a1=-3602879701896397/36028797018963968`. This gave unreadable names, and combined with bug 1 it gave wrong integer
   forms. The new helper `zead._frac` reads floats as the decimals they print as, so `a1=-0.1` gives exactly ZeaD
   6321. `in_7i_domain` is now evaluated in exact arithmetic.
3. Spec: the extent of the 7-instant domain was misstated; it is $p_1\in(-1.4,-0.6)$, $p_2\in(-1.0267,-0.8)$.
4. Naming: "6NgCD" changed to "6NτCD" as printed in [U19]. An unsourced AB citation was removed.

**Caveat added.** The `zead_7i_general` default $(-1,-1)$ lies near the edge of domain (16): the quadratic form equals
−6 there and the second root has modulus 0.976. Members with second-root modulus ≈ 0.83 exist, for example near
$(-1.32,-0.912)$. The robustness findings for this key in §6.2 therefore belong to the default and are not a property
of 7-instant formulas in general. The default was kept because it is the implementer's flagged ASSUMPTION, and
choosing an "optimal" member would be our own construction.
