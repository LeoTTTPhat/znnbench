# Spec: order reduction of the RK4 "non-causal reference" on ZNN

Owner files: `tests/test_rk4_order.py` and this file.
Code covered, not edited: `RK4` in `znnbench/discretizations/multistep.py` (key `rk4`).

## 0. Finding in one paragraph

The classical Runge–Kutta method of order 4 is the benchmark's non-causal reference integrator, registered as `rk4`. On linear ZNN its residual order depends on which quantity is held fixed as $\tau\to0$:

- **Fixed $h=\gamma\tau$** (the DTZNN convention): the residual is only **$O(\tau^2)$**.
- **Fixed $\gamma$** (the continuous-time convention): the residual is **$O(\tau^4)$**.

The reason is that linear ZNN is a Prothero–Robinson-type problem. Its stiffness parameter is $z=\tau\lambda=-h$, and at fixed $h$ this does not go to zero. RK4 has stage order 1 and weak stage order 1, and for that case [BKRSS25] gives an observed rate of $\min(p,q_w+1)=2$ at bounded $\tau L$. Its internal stage defects are $O(\tau^2)$, and a factor $z$ that does not shrink carries them into the local error. At fixed $h=0.3$ the error constant is $h^3/(96(1-R(-h)))=1.085\times10^{-3}$. Euler's is $1/(2h)=1.67$, so RK4 is about **1536 times smaller** at the same order. Consequence: at fixed $h$, every ZeaD formula with 4 or more instants has a higher residual order (3 to 6) than RK4. But RK4's constant is so small that it still has the smaller residual on any practical $\tau$ grid against the 4-instant formulas: at $h=0.3$ the crossover is at $\tau\approx10^{-4}$ to $10^{-3}$.

## 1. Sources (DOIs verified against their Crossref records, 2026-10-18)

| ref | citation | DOI | how it was checked |
|---|---|---|---|
| [PR74] | A. Prothero, A. Robinson, "On the stability and accuracy of one-step methods for solving stiff systems of ordinary differential equations", *Math. Comp.* 28(125) (1974) 145–162 | 10.1090/S0025-5718-1974-0331793-2 | Crossref record matches. The full text was not reachable (AMS returned 403). The test problem $y'=g'(x)+\lambda\{y-g(x)\}$ and the stiff-limit accuracy question ($h\,\mathrm{Re}\lambda\to-\infty$) are confirmed from the abstract as restated by secondary sources. |
| [HW96] | E. Hairer, G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd rev. ed., Springer Ser. Comput. Math. 14, 1996, Sect. IV.15 "B-Convergence" | 10.1007/978-3-642-05221-7 | Crossref record matches. **The book text was not accessible.** One secondary source confirms the pointer: arXiv:2505.15099 (Roberts, Shirokoff, Biswas, Seibold), Sect. 1.1, cites "[7, Section IV.15]" for the simplifying assumptions $B(q_1)$, $C(q_2)$ of B-convergence theory. The verifier checked this in the PDF text. **Unverified:** that IV.15 opens with the PR problem and RK local errors. This is from memory of the book, not from a source read here. The derivation below is self-contained and checked in exact arithmetic and numerically, so it does not depend on the book's wording. |
| [FSU81] | R. Frank, J. Schneid, C. W. Ueberhuber, "The concept of B-convergence", *SIAM J. Numer. Anal.* 18(5) (1981) 753–780 | 10.1137/0718051 | Crossref record matches (optional citation for where B-convergence originated). Note: 10.1137/0718050 is an unrelated erratum (Symes, 18(4)). |
| [BKRSS25] | A. Biswas, D. I. Ketcheson, S. Roberts, B. Seibold, D. Shirokoff, "Explicit Runge–Kutta methods that alleviate order reduction", *SIAM J. Numer. Anal.* 63(4) (2025) 1398–1426; arXiv:2310.02817 | 10.1137/23M1606812 | Crossref record and arXiv record match. Added by the verifier. **Full text read** on arXiv (HTML v2), for Sect. 1, Sect. 1.3 and eqs. (1)–(3). It gives: the stage-order residuals $\tau^{(k)}=\mathcal A c^{k-1}-c^k/k$; the weak stage order (WSO) $q_w$, i.e. the largest $q$ with $b^\top(I-z\mathcal A)^{-1}\tau^{(k)}=0$ for all $z$ and all $k\le q$ (eq. (3)); "explicit RK methods have stage order equal to (at most) one"; and the observed convergence rate $\min(p,q_w+1)$ for linear problems with time-dependent forcing or boundary data at fixed CFL, i.e. bounded $\Delta t\,L$. Classical methods (WSO 1) show second-order convergence in their Sect. 5. This is the explicit-method, bounded-$z$ result that our fixed-$h$ regime instantiates. |

**Scope relative to the sources.** [PR74] and [HW96] study *implicit* methods in the stiff limit $z\to-\infty$. Here the method is *explicit*, and $z=-h$ is fixed inside the stability interval. The algebra is the same: the recursion $e_{n+1}=R(z)e_n+\delta$ with stage defects. The verifier re-derived this recursion independently in sympy, from a symbolic RK4 step on a generic Taylor polynomial $\varphi$. Explicit RK methods with bounded $z$ are covered by [BKRSS25], whose rate $\min(p,q_w+1)$ predicts 2 for RK4. *Correction by the verifier:* the first version of this spec presented that regime as new, and it is not. The ZNN reading below (linear ZNN = PR problem with $z=-h$), the exact error polynomials, the constants and every number are ZNN-Bench derivations, flagged as such.

## 2. Linear ZNN is a Prothero–Robinson-type problem

For $E=A(t)x-b(t)$ with linear activation, ZNN is $A\dot x=-\gamma(Ax-b)-(\dot Ax-\dot b)$. With $x^*=A^{-1}b$ we have $\dot x^*=A^{-1}\dot b-A^{-1}\dot A x^*$. Then

$$
\dot x=-\big(\gamma I+M(t)\big)\big(x-x^*(t)\big)+\dot x^*(t),\qquad M=A^{-1}\dot A .
$$

- **Constant $A$** ($M=0$): this is exactly the vector PR problem $y'=\lambda(y-\varphi)+\varphi'$ with $\lambda=-\gamma$ and $\varphi=x^*$, one scalar PR problem per component. The test problem `ConstantAMovingB` (test-local) has this form. It reuses `linear_system`'s $b(t)$ and a `TVMatrix` frozen with `freq=0`, so $\dot A\equiv0$ exactly.
- **Time-varying $A$** (`linear_system`): the stiff matrix is $L(t)=-(\gamma I+M(t))$. $M$ is $O(1)$, while $\gamma=h/\tau\to\infty$ at fixed $h$. **ASSUMPTION (ZNN-Bench derivation, not in the sources):** to leading order the analysis below holds with $\tau L=-hI+O(\tau)$. Only the numerical prefactor check of Sect. 5 verifies it (0.12 % at $\tau=10^{-3}$).

The residual is $E=A(x-x^*)=Ae$.

## 3. RK4 on the PR problem: local error polynomials

Let $(\mathcal A,b,c)$ be the classical tableau: $c=(0,\tfrac12,\tfrac12,1)$, $b=(\tfrac16,\tfrac13,\tfrac13,\tfrac16)$, and $a_{21}=a_{32}=\tfrac12$, $a_{43}=1$. Write $z=\tau\lambda$ and $e_n=y_n-\varphi(t_n)$. Taylor-expand $\varphi$ in the stage equations. With the stage defects

$$
d_q=\frac{c^q-q\,\mathcal A c^{q-1}}{q!}\in\mathbb Q^4,
$$

the exact error recursion (cf. [HW96, IV.15]) is

$$
e_{n+1}=R(z)\,e_n+\delta_n,\qquad
\delta_n=\sum_{q\ge2}C_q(z)\,\tau^q\varphi^{(q)}(t_n),\qquad
C_q(z)=-\frac{1-q\,b^\top c^{q-1}}{q!}-z\,b^\top(I-z\mathcal A)^{-1}d_q .
$$

Here $R(z)=1+z+z^2/2+z^3/6+z^4/24$. Since $\mathcal A^4=0$, every $C_q$ is a polynomial in $z$. The two quantities to note are:

- **Stage order 1.** $d_1=0$, but $d_2=(0,\tfrac18,-\tfrac18,0)\neq0$. Every explicit RK method has stage order at most 1 ([BKRSS25], Sect. 1). The reason: let $i$ be the first stage with $c_i\neq0$. Stage order 2 would need $\sum_ja_{ij}c_j=c_i^2/2$, but the left side is 0 because every earlier $c_j=0$. For RK4, $i=2$.
- **Link to [BKRSS25].** $d_q=-\tau^{(q)}/(q-1)!$ with $\tau^{(q)}=\mathcal Ac^{q-1}-c^q/q$. The $z$-dependent part of $C_q$ is therefore $z\,b^\top(I-z\mathcal A)^{-1}\tau^{(q)}/(q-1)!$, and $C_q-C_q(0)\equiv0$ for $q\le q_w$. For RK4, $b^\top\mathcal A^j\tau^{(2)}=(0,0,-\tfrac1{96},0)_j$, so $q_w=1$. This is tested.
- **Exact polynomials** (computed with `Fraction` in the test):

| $q$ | $C_q(z)$ | at fixed $\gamma$ ($z=-\gamma\tau$): contribution to $\delta$ |
|---|---|---|
| 1 | $0$ | — |
| 2 | $-z^3/96$ | $\gamma^3\tau^5\varphi''/96$ |
| 3 | $z^2/288-z^3/576$ | $\gamma^2\tau^5\varphi'''/288+O(\tau^6)$ |
| 4 | $-z/576+z^2/1152-z^3/4608$ | $\gamma\tau^5\varphi^{(4)}/576+O(\tau^6)$ |
| 5 | $1/2880-z/1440+z^2/7680-z^3/46080$ | $\tau^5\varphi^{(5)}/2880+O(\tau^6)$ (classical order-4 error term) |

The order conditions make $b^\top\mathcal A^kd_2=0$ for $k=0,1$. The first non-zero term is $b^\top\mathcal A^2d_2=b_4a_{43}a_{32}c_2^2/2=1/96$. Hence $C_2(z)=-z^3/96$.

## 4. The two regimes

**Fixed $h$** ($z=-h$ fixed, $|R(-h)|<1$ for $h<2.785$). Then $\delta_n=C_2(-h)\tau^2\ddot x^*(t_n)+O(\tau^3)$, with $C_2(-h)=h^3/96\ne0$. The error contracts by the fixed factor $R(-h)$, and $\delta_n$ changes by $O(\tau^3)$ per step. So the forced response is quasi-static:

$$
e_{\rm ss}(t)=\frac{h^3}{96\,(1-R(-h))}\,\tau^2\,\ddot x^*(t)+O(\tau^3),\qquad
E_{\rm ss}=A\,e_{\rm ss}\ \Rightarrow\ \text{MSSRE}=\kappa_{\rm RK4}(h)\,\tau^2\max_{t}\|A\ddot x^*\|\,(1+O(\tau)).
$$

The constant is $\kappa_{\rm RK4}(h)=h^3/(96(1-R(-h)))$, about $h^2/96$ for small $h$. Euler's look-ahead gives $e_{\rm ss}=-\tau^2\ddot x^*/(2h)$, so $\kappa_{\rm Euler}(h)=1/(2h)$. Both have **order 2**, and the ratio is $\kappa_{\rm Euler}/\kappa_{\rm RK4}=48(1-R(-h))/h^4$:

| $h$ | $\kappa_{\rm RK4}(h)$ | $\kappa_{\rm Euler}(h)$ | ratio |
|---|---|---|---|
| 0.3 | $1.0852\times10^{-3}$ | 1.667 | 1535.8 |
| 0.05 | $2.670\times10^{-5}$ | 10 | $3.75\times10^{5}$ |

**Fixed $\gamma$** ($z\to0$). Every term of $\delta$ is $O(\tau^5)$ (table above), and $1-R(z)=\gamma\tau+O(\tau^2)$. The error is the consistent discretisation of $\dot e=-\gamma e+\delta/\tau$, so for constant $A$

$$
e_{\rm ss}=\tau^4\Big(\tfrac{\mathrm d}{\mathrm dt}+\gamma\Big)^{-1}\Big[\tfrac{\gamma^3}{96}\ddot x^*+\tfrac{\gamma^2}{288}x^{*(3)}+\tfrac{\gamma}{576}x^{*(4)}+\tfrac{1}{2880}x^{*(5)}\Big]+O(\tau^5),
$$

which is **order 4**. For time-varying $A$, $M$ enters the filter, but the order stays at 4, the classical order. For $\gamma\gg\omega$ (data frequency) the bracket is dominated by $\gamma^2\tau^4\ddot x^*/96=h^2\tau^2\ddot x^*/96$. This is the fixed-$h$ term, so the two regimes join continuously.

**Why the regimes differ.** Write $p$ for the classical order and $q_s$ for the stage order; for RK4, $p=4$ and $q_s=1$. The stage defect $O(\tau^{q_s+1})$ reaches the local error multiplied by powers of $z$. When $z\to0$ (fixed $\gamma$) those powers supply the missing $\tau$'s. When $z$ is fixed (fixed $h$) they do not. This is the same mechanism as [PR74]/[HW96] order reduction, but in the regime $z$ bounded and non-zero instead of $z\to-\infty$.

The same argument applies to every explicit method with $2\le s=p\le4$. Then $b^\top\mathcal A^{s-2}d_2=b_sa_{s,s-1}\cdots a_{32}c_2^2/2\neq0$, because $b^\top\mathcal A^{s-2}c=b_sa_{s,s-1}\cdots a_{32}c_2=1/s!\ne0$ forces that product to be non-zero. The order conditions make the lower coefficients vanish: $b^\top\mathcal A^kd_2=0$ for $k\le s-3$, and the constant term is 0 for $p\ge2$. So $C_2$ is the monomial
$$C_2(z)=-\frac{c_2}{2\,s!}\,z^{s-1}\qquad(\text{RK4: }-z^3/96;\ \text{3/8 rule: }-z^3/144;\ \text{Kutta-3: }-z^2/24;\ \text{Heun-2: }-z/4).$$
It has no root in $h>0$, so the fixed-$h$ residual order is 2 for every such method. This is a ZNN-Bench derivation. The verifier confirmed it symbolically and tested it exactly on six tableaux (`test_explicit_s_equals_p_methods_reduce_to_order_two_at_fixed_h`); only `rk4` is registered. A corollary: every order-4 method with $c=(0,\tfrac12,\tfrac12,1)$ (a one-parameter family) has the same $C_2$ and the same fixed-$h$ constant. On *linear* problems, including time-varying $L(t)$, these methods even give identical steps. The verifier checked this numerically for the member $b_3=\tfrac14$: the difference from classical RK4 is $6\times10^{-17}$ on linear problems and $3\times10^{-8}$ on a nonlinear one. That is why the tableau test reads $(\mathcal A,b,c)$ off the stage inputs rather than relying on $R(z)$ and the stage times alone.

**Contrast with look-ahead (ZeaD) formulas.** A look-ahead LMM evaluates the field only at the grid sample $(t_k,x_k)$, so $x^*$ has no stage defect. Its defect is the truncation error $C_{q+1}\tau^{q+1}x^{*(q+1)}$, independent of $\gamma$. That is why fixed $h$ *gains* one order over fixed $\gamma$ for LMMs ($q+1$ versus $q$). For RK4 the situation is reversed: fixed $h$ *loses* two orders (2 versus 4).

## 5. Tests (`tests/test_rk4_order.py`, 24 tests, about 16 s)

All runs use linear `znn` with $n=4$, $\kappa=10$, seed 1, $T=4$, and the MSSRE over the last 25 % of the horizon. They start at $x_0=x^*(0)$.

**ASSUMPTION (protocol):** starting on the solution removes the initial transient, so the steady state measured is driven by truncation error alone. At $h=0.05$, $\tau=10^{-2}$ the gain is only 5. A random start then leaves $\approx\|E_0\|e^{-15}\approx3\times10^{-6}$ in the steady window, which hides the $5\times10^{-8}$ steady residual: the measured slope becomes 3.8 instead of 2. At $h=0.3$ and $\gamma=10$ a random start gives the same numbers.

| test | what it checks | result |
|---|---|---|
| `test_registered_rk4_is_the_classical_tableau` | registered `rk4`. The full Butcher tableau $(\mathcal A,b,c)$ is read off the stage inputs (stage $j$ returns $e_j$), for $x$ and for $z$, to $10^{-14}$. One step on $\dot x=\lambda x$ equals $R(z)$ to $10^{-14}$. Stage times are $t+c_i\tau$, and only stage 1 receives $k$ (non-causal). Also checks `causal=False`, and that `expected_order` is `None` or 2, the fixed-$h$ convention of `test_discretizations.py`. Optional `expected_order_fixed_h` and `expected_order_fixed_gamma` attributes are checked against 2 and 4 when present. | pass |
| `test_constant_A_problem_is_consistent` | $\dot A\equiv0$; $E(x^*)=0$; $\partial_tE$ vs. finite differences | pass |
| `test_rk4_stage_order_one_and_reduction_polynomial` | exact: the 8 order conditions; $d_1=0$, $d_2=(0,\tfrac18,-\tfrac18,0)$; the full polynomials $C_1,\dots,C_5$ of Sect. 3; $C_2(-h)/(1-R(-h))=\kappa_{\rm RK4}(h)$ | pass |
| `test_rk4_weak_stage_order_is_one` | exact: $d_k=-\tau^{(k)}/(k-1)!$ for $k\le5$; $b^\top\mathcal A^j\tau^{(1)}=0$ and $b^\top\mathcal A^j\tau^{(2)}=(0,0,-\tfrac1{96},0)$, i.e. WSO $=1$ ([BKRSS25] eq. (3)) | pass |
| `test_explicit_s_equals_p_methods_reduce_to_order_two_at_fixed_h[*]` | exact, 6 tableaux (Heun-2, Ralston-2, Kutta-3, Heun-3, RK4, 3/8 rule): order $p=s$, stage order 1, $C_2(z)=-c_2z^{s-1}/(2\,s!)$ | pass |
| `test_prothero_robinson_local_error_matches_expansion` | one registered-RK4 step on the scalar PR problem ($\varphi=\sin1.3t$) equals $\sum_{q=2}^{11}C_q(z)\tau^q\varphi^{(q)}$ to relative $10^{-6}$ (round-off limited). Local-error slope at fixed $z=-0.3$: 2.0, with leading term $h^3\tau^2\varphi''/96$ matched to 1 %. Slope at fixed $\lambda=-10$: 5.0. | pass |
| `test_rk4_residual_order[*]` | MSSRE slopes (table below); tolerance 0.25, and the last local slope within 0.15 | pass |
| `test_const_A_mssre_equals_exact_prothero_robinson_prediction[*]` | constant $A$, no asymptotics: MSSRE equals the exact discrete steady state $\mathrm{Im}[a_ie^{i(w_it+\phi_i)}\sum_qC_q(z)(iw_i\tau)^q/(e^{iw_i\tau}-R(z))]$ in all three regimes (observed agreement $\le1.6\times10^{-6}$ relative; test tolerance $2\times10^{-5}$) | pass |
| `test_error_constant_vs_euler_at_h03[*]` | Euler/RK4 MSSRE ratio $=1535.8$ within 2 % for $\tau\le10^{-2}$; absolute prefactors $\kappa(h)\tau^2\max\|A\ddot x^*\|$ within 1 % at $\tau=10^{-3}$; Euler slope 2 | pass |
| `test_zead_beats_rk4_in_order_not_in_constant[*]` | at $h=0.3$: Taylor–Zhang slope $3.0>2.0$, yet its MSSRE is larger than RK4's at every $\tau\in\{3\cdot10^{-2},\dots,10^{-3}\}$; `zead_8i_wu2020` beats RK4 by more than 10× at $\tau=10^{-2}$ | pass |

Measured RK4 MSSRE slopes (least squares; local slopes in brackets):

| regime | $\tau$ grid | `linear_system` | constant $A$ |
|---|---|---|---|
| $h=0.3$ | 3e-2, 1e-2, 3e-3, 1e-3 | 2.002 [2.015, 1.997, 1.998] | 1.997 [1.992, 1.999, 2.000] |
| $h=0.05$ | 1e-2, 5e-3, 2.5e-3, 1.25e-3 | 2.034 [2.094, 2.016, 1.998] | 1.983 [1.960, 1.990, 1.997] |
| $\gamma=10$ | 5e-2, 2.5e-2, 1.25e-2, 6.25e-3 | 4.103 [4.177, 4.091, 4.046] | 4.099 [4.170, 4.087, 4.044] |

Prefactor checks at $h=0.3$:

- Euler/RK4 MSSRE ratio. `linear_system`: 1508 / 1546 / 1541 / 1538 at $\tau$ = 3e-2 / 1e-2 / 3e-3 / 1e-3. Constant $A$: 1542 / 1536.5 / 1535.8 / 1535.8. Predicted: 1535.8.
- Measured/predicted absolute prefactor at $\tau=10^{-3}$. RK4: 0.9988 (`linear_system`) and 0.99999 (constant $A$). Euler: 1.00003 and 0.99999.

## 6. Reading of `expected_order=None` in `multistep.py`

`RK4.expected_order` is `None`, and `docs/ZOO.md` shows "—" in the order column. **Read this as "2 at fixed $h$, 4 at fixed $\gamma$" on linear ZNN.** No single number is correct. The fixed-$h$ order-verification protocol of `docs/DESIGN.md` would report 2, which says nothing about RK4's classical order. `multistep.py` is not edited here, because this area does not own it. If the owner wants the value machine-readable, a non-breaking option is an attribute such as `expected_order_fixed_h = 2` / `expected_order_fixed_gamma = 4`. That is a suggestion only; nothing in the benchmark relies on it. Setting `expected_order = 2` would also be consistent, because `expected_order` follows the fixed-$h$ convention of `tests/test_discretizations.py::test_empirical_order_fixed_h`. Setting it to 4 would be wrong. The test accepts `None` or 2, and checks the optional attributes when they exist, so the multistep owner can choose either without breaking this file. (Verifier change: the first version asserted `expected_order is None` exactly.)

## 7. Implications for order plots and references

1. **RK4 is not an "order-4 reference" in fixed-$h$ order plots.** In the DTZNN convention its residual slope is 2, the same as Euler. Every ZeaD formula with 4 or more instants has a higher residual order: 3 for 4 instants, 4 for 5–6, 5 for 7–8, 6 for 10–11. An order plot at fixed $h$ that includes RK4 must say so. Otherwise RK4 looks like a second-order method.
2. **But RK4's constant is much smaller.** At equal $\tau$ and $h$ it has the smaller residual until a crossover step $\tau_c$. With a single harmonic of unit frequency ($|x^{*(m)}|\propto\omega^m$, $\omega=1$; **ASSUMPTION** for this estimate), $|C_{q+1}|\tau^{q+1}/h=\kappa_{\rm RK4}(h)\tau^2$ gives $\tau_c=(\kappa_{\rm RK4}(h)\,h/|C_{q+1}|)^{1/(q-1)}$:

| key | instants | residual order | $h_{\max}$ | $\lvert C_{q+1}\rvert$ | $\tau_c$ at $h=0.3$ (analytic) | measured $\tau_c$ at $h=0.3$: `linear_system` / const-$A$ | $\tau_c$ at $h=0.05$ (analytic) |
|---|---|---|---|---|---|---|---|
| `taylor_zhang` | 4 | 3 | 1.000 | 0.6667 | 4.9e-4 | 1.4e-4* / 4.1e-4* | 2.0e-6 |
| `zead_4i_general` | 4 | 3 | 2.000 | 0.4167 | 7.8e-4 | 2.3e-4* / 6.5e-4* | 3.2e-6 |
| `zead_4i_6321` | 4 | 3 | 0.800 | 0.2667 | 1.2e-3 | 3.5e-4* / 1.0e-3 | 5.0e-6 |
| `zead_4i_fifd` | 4 | 3 | 1.000 | 0.2917 | 1.1e-3 | 3.2e-4* / 9.3e-4* | 4.6e-6 |
| `zead_5i`, `zead_5i_general`, `uhlig_2_3` | 5 | 4 | 0.281 | 0.1944 | unstable | — | 2.6e-3 |
| `zead_5i_mahuang2025` | 5 | 4 | 0.172 | 0.2115 | unstable | — | 2.5e-3 |
| `zead_6i_guo2017` | 6 | 4 | 0.333 | 0.3125 | 3.2e-2 | 8.6e-3 / 1.6e-2 | 2.1e-3 |
| `zead_6i_qiu2019` | 6 | 4 | 0.667 | 0.375 | 2.9e-2 | 7.9e-3 / 2.3e-2 | 1.9e-3 |
| `uhlig_3_4` | 7 | 5 | 0.054 | 0.2711 | unstable | — | 1.7e-2 |
| `zead_7i_general` | 7 | 5 | 0.164 | 0.2308 | unstable | — | 1.8e-2 |
| `zead_7i_cang2024` | 7 | 5 | 0.116 | 0.2601 | unstable | — | 1.7e-2 |
| `zead_8i_chen2019` | 8 | 5 | 0.428 | 0.4882 | 8.7e-2 | 2.7e-2 / 5.5e-2 | 1.4e-2 |
| `zead_8i_wu2020` | 8 | 5 | 0.519 | 0.4414 | 9.0e-2 | 2.8e-2 / 5.9e-2 | 1.4e-2 |
| `uhlig_4_6` | 10 | 6 | 0.176 | 0.4015 | unstable | — | 4.3e-2 |
| `zead_10i_wu2020` | 10 | 6 | 0.129 | 0.6725 | unstable | — | 3.8e-2 |
| `zead_11i_chen2023` | 11 | 6 | 0.223 | 0.84 | unstable | — | 3.6e-2 |

\* Extrapolated from $\tau=10^{-3}$ using ratio $\propto\tau^{q-1}$, because the crossover lies below the measured grid. Other measured values are log-log interpolated on a 21-point grid from 0.1 to 1e-3. These were exploration runs and are not part of the test file; the test asserts only the Taylor–Zhang and `zead_8i_wu2020` facts. The measured $\tau_c$ is 1–3.5× smaller than the single-harmonic estimate, because the multi-frequency data (and, for `linear_system`, $A(t)$) have relatively larger high derivatives. The $h=0.05$ crossovers of the 7–11-instant formulas fall at $\tau\approx10^{-2}$, where $\gamma=h/\tau\approx1$–5 is not large compared with $\omega$, so the quasi-static estimate is only indicative there.

   The table now covers every registered look-ahead key (`LookAhead` subclasses with the generic step), as of the verification on 2026-10-18. The verifier added the rows for `zead_5i_mahuang2025` and `zead_7i_cang2024`, which landed concurrently, from their registered coefficients. Both are unstable at $h=0.3$. Recompute any row with $\tau_c=(\kappa_{\rm RK4}(h)\,h/|C_{q+1}|)^{1/(q-1)}$, where $C_{q+1}$ = `taylor_moments(coeffs)[q+1]` from `zead.py`. Non-look-ahead causal schemes (`ab2`–`ab4`, `cdtznn_mahuang2025`, the `consistent` and prediction–correction families) are outside this table.

   Reading of the table: against the 4-instant formulas, RK4 wins on every practical grid ($\tau\ge10^{-3}$ at $h=0.3$; all $\tau\gtrsim10^{-5}$ at $h=0.05$). The 6–8-instant formulas overtake it at $\tau\sim10^{-2}$.
3. **Cost and causality.** These comparisons are at equal $\tau$. RK4 uses 4 field evaluations, i.e. 4 linear solves, per step against 1 for a look-ahead formula. It also needs the problem data at $t_k+\tau/2$ and $t_k+\tau$, which a real-time (causal) DTZNN does not have. A cost-matched comparison would pit RK4 at step $\tau$ against a look-ahead formula at $\tau/4$. That also changes $h$, so it cannot be read off the fixed-$h$ tables. RK4 remains a *reference*, not a competitor.
4. **The fixed-$h$/fixed-$\gamma$ rule for look-ahead formulas (residual order $q+1$ at fixed $h$, $q$ at fixed $\gamma$) is LMM-specific.** The consequence "both regimes share the leading term $\tau^{q+1}/h=\tau^q/\gamma$" holds for look-ahead formulas. It does not hold for multistage methods:
   > "The extra order at fixed $h$ is specific to formulas that evaluate the vector field only at the sample $(t_k,x_k)$. Classical RK4, the usual non-causal reference, has stage order 1. At fixed $h$ its residual is only $O(\tau^2)$, with constant $h^3/(96(1-R(-h)))\approx h^2/96$, against $O(\tau^4)$ at fixed $\gamma$. This is the order reduction of Prothero and Robinson's stiff test problem [PR74; HW96, Sect. IV.15], of which linear ZNN is an instance with stiffness $z=-h$. For explicit methods at bounded $z$ the observed rate is $\min(p,q_w+1)$ with weak stage order $q_w=1$ [BKRSS25]."
   (Verifier note: [BKRSS25] is the only one of these references whose full text was read.)
5. **Using RK4 as a "continuous-time" reference.** To approximate the continuous ZNN trajectory (as in the model tests that check $\|E(t)\|=\|E_0\|e^{-\gamma t}$), run RK4 at fixed $\gamma$ with small $\tau$; its error is then $O(\tau^4)$. At fixed $h$ the RK4 residual floor $\kappa_{\rm RK4}(h)\tau^2\|A\ddot x^*\|$ is a discretisation artifact, since the continuous ZNN residual from $x_0=x^*(0)$ is exactly 0. It should not be reported as a property of the model.

## 8. Assumptions and limitations (summary)

- **ASSUMPTION (Sect. 2):** the time-varying-$A$ extension ($\tau L=-hI+O(\tau)$) is a leading-order argument. It is verified numerically, not proven here. The measured/predicted prefactor on `linear_system` at $h=0.3$ is 0.9928, 0.9968, 0.99885 and 0.99965 at $\tau$ = 1e-2, 3e-3, 1e-3 and 3e-4, so the deviation shrinks like $O(\tau)$ (verifier run).
- **ASSUMPTION (Sect. 5):** $x_0=x^*(0)$ in all order runs, which isolates the truncation-driven steady state.
- **ASSUMPTION (Sect. 7):** the analytic crossover estimates use one harmonic with $\omega=1$. The measured crossovers are given alongside.
- **Linear activation only.** For a smooth activation the fixed-$h$ result holds with $h\to\Phi'(0)h$ (linearisation at $E=0$). This is not tested here. Sign-type activations are not covered.
- **Sources:** the [PR74] and [HW96] texts were not consulted directly (see Sect. 1). The [BKRSS25] full text was read, and it supports the stage-order and WSO facts and the rate of 2 for RK4 at bounded $z$. The derivation is self-contained and checked in exact arithmetic and against simulation to round-off.

## 9. Verification (adversarial review, 2026-10-18)

The verifier recomputed everything with independent scratch scripts (kept outside the repository) and did not reuse the test helpers. **Verdict: the finding is correct.** No equation, coefficient or reported number was wrong. Changes were limited to test strength, source coverage and one overstated novelty claim.

**Independent recomputations**

| item | method | result |
|---|---|---|
| $R(z)$, $C_0,\dots,C_7$ | sympy. One classical RK4 step on $y'=\lambda(y-\varphi)+\varphi'$ with $\varphi$ a generic degree-9 Taylor polynomial and $\lambda=z/\tau$; coefficients of $e_n$ and of each $\varphi^{(q)}(t_n)$ extracted. This does **not** use the stage-defect formula. | $R=1+z+\dots+z^4/24$. $C_1=0$, $C_2=-z^3/96$, $C_3=z^2/288-z^3/576$, $C_4=-z/576+z^2/1152-z^3/4608$, $C_5=1/2880-z/1440+z^2/7680-z^3/46080$: all match Sect. 3. $C_5(0)=1/2880$ is Simpson's error constant, as it should be. |
| $C_q$ formula and recursion | hand derivation, with $E_i=Y_i-\varphi(t_n+c_i\tau)$ | gives $e_{n+1}=R(z)e_n+\sum_q\tau^q\varphi^{(q)}[-(1-q\,b^\top c^{q-1})/q!-z\,b^\top(I-z\mathcal A)^{-1}d_q]$, identical to Sect. 3. |
| $\kappa_{\rm RK4}$, ratio | sympy, exact rationals | $\kappa(0.3)=1.0852265\times10^{-3}$, ratio $1535.778$; $\kappa(0.05)=2.66981\times10^{-5}$, ratio $3.74558\times10^5$; $\lvert R(-h)\rvert<1$ for $h<2.78529$. |
| general $s=p$ claim | sympy on 6 tableaux | $C_2=-c_2z^{s-1}/(2\,s!)$ in all cases; now a test. |
| MSSRE slopes, both problems, 3 regimes | own RK4 and Euler loops written from scratch (no `simulate`, no registered discretizer) | All 18 slopes and local slopes of Sect. 5 reproduced to the printed digits, e.g. 2.002 [2.015, 1.997, 1.998], 4.099 [4.170, 4.087, 4.044]. |
| Euler/RK4 ratios and prefactors at $h=0.3$ | same loops | `linear_system` 1508.4 / 1545.9 / 1540.7 / 1537.6; constant $A$ 1542.2 / 1536.5 / 1535.8 / 1535.8. RK4 prefactor at $\tau=10^{-3}$: 0.99885 and 0.99999. All match. |
| random-start protocol claim | `simulate` with `x0=None` | $h=0.05$: slope 3.802, MSSRE at $\tau=10^{-2}$ is $2.80\times10^{-6}$ against $4.72\times10^{-8}$ from $x^*(0)$. $h=0.3$: identical numbers. Confirms the ASSUMPTION note. |
| analytic crossover table | own Taylor moments, `lookahead_stability` for $h_{\max}$, every registered `LookAhead` key | All 14 published rows reproduced ($h_{\max}$, $\lvert C_{q+1}\rvert$, $\tau_c$ at 0.3 and 0.05). Two rows added for the keys that landed concurrently. |
| measured crossovers at $h=0.3$ | `simulate` on the same 21-point grid, plus extrapolation from $10^{-3}$ | Taylor–Zhang 1.42e-4\* / 4.07e-4\*; `zead_4i_general` 2.26e-4\* / 6.51e-4\*; `zead_4i_6321` 3.54e-4\* / 1.02e-3; `zead_4i_fifd` 3.23e-4\* / 9.30e-4\*; `zead_6i_guo2017` 8.6e-3 / 1.64e-2; `zead_6i_qiu2019` 7.9e-3 / 2.34e-2; `zead_8i_chen2019` 2.67e-2 / 5.52e-2; `zead_8i_wu2020` 2.76e-2 / 5.94e-2. All match the table. |
| DOIs | Crossref records | [PR74], [HW96], [FSU81] and [BKRSS25] match; 10.1137/0718050 is indeed an unrelated erratum. |

**Source re-check.** The AMS PDF and abstract pages of [PR74] returned HTTP 403 to the verifier as well, and [HW96] is not openly accessible. So the IV.15 pointer remains secondary-confirmed only (Sect. 1). To compensate, the verifier added [BKRSS25], a primary source whose full text is open on arXiv. It states the explicit-RK stage-order bound, defines the weak stage order, and gives the observed rate $\min(p,q_w+1)$ at bounded $\Delta t\,L$. The fixed-$h$ order 2 of RK4 therefore follows from a published, readable result, independently of the ZNN-Bench derivation.

**Weaknesses found and fixed (in the owned files only)**

1. *The tableau test did not identify RK4.* It checked only $R(z)$ and the stage times, and every member of the one-parameter family of order-4 methods with $c=(0,\tfrac12,\tfrac12,1)$ passes those. A mutation run showed that the member $b_3=\tfrac14$ passed **every** original test, because on linear problems it gives the same steps as RK4 to round-off. The test now reads the full $(\mathcal A,b,c)$ off the stage inputs, for $x$ and $z$; the mutant fails it. The explicit-midpoint mutant fails 5 of the 6 tests probed; only the fixed-$h$ slope test passes it, as expected, since both methods have order 2 at fixed $h$.
2. *A brittle cross-area assertion.* `expected_order is None` would break if the `multistep.py` owner adopted the spec's own suggestion. The test now accepts `None` or 2, the fixed-$h$ convention, and checks the optional regime attributes when present.
3. *Partial exact checks.* $C_4$ and $C_5$ were checked only in their lowest coefficient. They are now checked in full.
4. *Untested claim.* The general explicit $s=p$ statement of Sect. 4 was a proof sketch with "only RK4 tested". It now has a closed form, $C_2=-c_2z^{s-1}/(2\,s!)$, tested exactly on six tableaux.
5. *Overstated novelty.* Sect. 1 presented the bounded-$z$ explicit regime as a ZNN-Bench derivation. It is covered by [BKRSS25]. This is corrected, and the citation is added to the drafted paper sentence.
6. *Outdated table.* The crossover table now includes `zead_5i_mahuang2025` and `zead_7i_cang2024`.

No registry keys belong to this area, and nothing is registered without a primary source. `ConstantAMovingB` is test-local and deliberately unregistered.

**Remaining caveats (not fixable in these files)**

- [PR74] and [HW96] were not read directly.
- The time-varying-$A$ extension is supported numerically ($O(\tau)$ convergence of the prefactor), not proven (Sect. 8).
- The measured crossover columns come from exploration runs; only the Taylor–Zhang and `zead_8i_wu2020` facts are asserted in the test file.
- `docs/ZOO.md` (not owned here) still lacks rows for `zead_5i_mahuang2025` and `zead_7i_cang2024`. This makes `tests/test_integration.py::test_zoo_catalogue_covers_every_registered_key` fail, independently of this area.
