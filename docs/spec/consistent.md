# Area `consistent`: consistent discretizations of finite-, fixed- and predefined-time ZNN

**Files:** `znnbench/discretizations/consistent.py`, `tests/test_consistent.py` (116 tests, about 17 s, peak RSS about 0.1 GB).

**Registered discretization keys** (all causal, `implicit = True`):

| key | look-ahead formula | error map | expected residual order (fixed $h=\gamma\tau$) |
|---|---|---|---|
| `peb_euler` | Euler | homogeneity-based (PEB 2019) | 2 |
| `exact_flow_euler` | Euler | exact flow / settling-time coordinate (JR 2020) | 2 |
| `semi_implicit_euler` | Euler | frozen-coefficient linearly implicit | 2 |
| `implicit_taylor_zhang` | Taylor–Zhang | resolvent (backward Euler), **error memory** | 3 |
| `peb_taylor_zhang` | Taylor–Zhang | homogeneity-based, error memory | 3 |
| `exact_flow_taylor_zhang` | Taylor–Zhang | exact flow, error memory | 3 |
| `semi_implicit_taylor_zhang` | Taylor–Zhang | frozen coefficient, error memory | 3 |
| `implicit_taylor_zhang_naive` | Taylor–Zhang | resolvent, **no** error memory (ablation only) | 3 for linear Φ; lost for sign-type Φ |
| `consistent` | factory `consistent(formula, error_map, memory=True)`. `formula` is `'euler'`, `'taylor_zhang'` or **any registered `LookAhead` key** (for example the ZeaD formulas of the `zead` area). | | that of the formula |

The backward-Euler/Euler combination already exists as `implicit_euler_err` in `multistep.py`.

Scheme names (`d.name`) equal the registry keys, so runner tables show e.g. `exact_flow_euler` (fixed in verification; before, the exact-flow schemes were named `exact_euler`/`exact_taylor_zhang`).

**Registered model key:** `znn_consistent` (`ConsistentZNN`). It is a `DesignFormulaZNN` that honours the error-map request protocol (§6). **Local activation keys** `signed_power_sum` (`PowerSum`) and `jr_predefined` (`PredefinedTimeJR`) are resolvable through `consistent.get_activation(key, **kw)` and `make_model("znn_consistent", activation=key, ...)`. They are deliberately **not** inserted into the shared `activations.REGISTRY` (see §11, finding F2): `tests/test_activations.py` iterates that registry and expects an independent reference formula for every key. The key `power_sum` belongs to the Yang–Zhang activation $\sum_k e^{2k-1}$ of `activations.py`.

---

## 1. Sources

| tag | reference | what was consulted |
|---|---|---|
| [PEB19] | A. Polyakov, D. Efimov, B. Brogliato, "Consistent discretization of finite-time and fixed-time stable systems", *SIAM J. Control Optim.* 57(1):78–103, 2019. doi:10.1137/18M1197345 | Full text of HAL preprint hal-01838712**v1** (July 2018). **ASSUMPTION:** the equation numbers below follow that preprint. The published version (from HAL v2, Nov 2018) may number them differently, but the abstract is identical. |
| [JR20] | E. Jiménez-Rodríguez, R. Aldana-López, J. D. Sánchez-Torres, D. Gómez-Gutiérrez, A. G. Loukianov, "Consistent discretization of a class of predefined-time stable systems", *IFAC-PapersOnLine* 53(2):628–633, 2020. doi:10.1016/j.ifacol.2020.12.806 | Full text of arXiv:2001.08318v1. Equation numbers are the arXiv ones. |
| [AL19] | R. Aldana-López, D. Gómez-Gutiérrez, E. Jiménez-Rodríguez, J. D. Sánchez-Torres, M. Defoort, "Enhancing the settling time estimation of a class of fixed-time stable systems", *Int. J. Robust Nonlinear Control* 29(12):4135–4148, 2019. doi:10.1002/rnc.4600 | arXiv:1809.07012 (Theorem 1) |
| [AB10] | V. Acary, B. Brogliato, "Implicit Euler numerical scheme and chattering-free implementation of sliding mode systems", *Syst. Control Lett.* 59(5):284–293, 2010. doi:10.1016/j.sysconle.2010.03.002 | Abstract only. The principle (implicit Euler on the set-valued sign gives exact, chattering-free convergence) is restated in [PEB19, §1.1] and [JR20, §3.2]. |
| [HAB16] | O. Huber, V. Acary, B. Brogliato, "Lyapunov stability and performance analysis of the implicit discrete sliding mode control", *IEEE TAC* 61(10):3016–3030, 2016. doi:10.1109/TAC.2015.2506991 | Bibliographic record only. **Not read in full.** |
| [LCL13] | S. Li, S. Chen, B. Liu, "Accelerating a recurrent neural network to finite-time convergence for solving time-varying Sylvester equation by using a sign-bi-power activation function", *Neural Process. Lett.* 37(2):189–205, 2013. doi:10.1007/s11063-012-9241-1 | Activation as implemented in `activations.py`. |
| [Pol12] | A. Polyakov, "Nonlinear feedback design for fixed-time stabilization of linear control systems", *IEEE TAC* 57(8):2106–2110, 2012. doi:10.1109/TAC.2011.2179869 | Bibliographic record. The bi-power fixed-time field and its settling bound are restated in [AL19, Eq. (1)–(2)]. |
| [ZDQ17] | Y. Zhang, Y. Ding, B. Qiu, Y. Zhang, X. Li, "Signum-function array activated ZNN with easier circuit implementation and finite-time convergence for linear systems solving", *Inf. Process. Lett.* 124:30–34, 2017. doi:10.1016/j.ipl.2017.04.008 | Bibliographic record. Justifies the pure sign activation as a ZNN activation. |
| [ST18] | J. D. Sánchez-Torres, D. Gómez-Gutiérrez, E. López, A. G. Loukianov, "A class of predefined-time stable dynamical systems", *IMA J. Math. Control Inf.* 35(Suppl 1):i1–i29, 2018. doi:10.1093/imamci/dnx004 | Cited through [JR20]. **Not read.** |
| [TZ15] | Y. Zhang, L. Jin, D. Guo, Y. Yin, Y. Chou, "Taylor-type 1-step-ahead numerical differentiation rule for first-order derivative approximation and ZNN discretization", *J. Comput. Appl. Math.* 273:29–40, 2015. doi:10.1016/j.cam.2014.05.027 | Coefficients taken from `multistep.taylor_zhang` (owned by another area). The docstring there cites "Zhang, Gong, Yang, Li, TNNLS 2018". **Flagged for the multistep owner:** the Taylor-type formula itself dates from [TZ15]. |
| [SMK16] | A. Simonetto, A. Mokhtari, A. Koppel, G. Leus, A. Ribeiro, "A class of prediction-correction methods for time-varying convex optimization", *IEEE TSP* 64(17):4576–4591, 2016. doi:10.1109/TSP.2016.2568161 | Related work only (§4.4). |

---

## 2. Setting: every ZNN discretization is a one-step error map

The design-formula ZNN imposes, element-wise, the error dynamics

$$\dot E(t) = -\gamma\,\Phi(E(t)),\qquad J(x,t)\,\dot x = -\gamma\Phi(E) - \partial_t E(x,t). \tag{C1}$$

A causal one-step-ahead (look-ahead) formula
$\tau\dot x_k \approx c_{-1}x_{k+1} + \sum_{j=0}^{s-1}c_j x_{k-j}$, with $\sum_{j\ge -1}c_j = 0$, gives the update

$$x_{k+1} = \Big(\tau\dot x_k - \sum_{j=0}^{s-1} c_j x_{k-j}\Big)\Big/c_{-1},\qquad J_k\dot x_k = r_k - \partial_t E_k, \tag{C2}$$

where $r_k$ is the desired error rate supplied by the model. The explicit choice is $r_k=-\gamma\Phi(E_k)$.

**Lemma 1 (error-map reduction).** Let $E(x,t)=Ax-b$ with constant $A$ and $b$ (a time-invariant linear system, so $\partial_tE\equiv0$), and let the formula be Euler ($c=(1,-1)$). Then (C2) gives exactly

$$E_{k+1} = E_k + \tau r_k .$$

*Proof.* $E_{k+1}=A(x_k+\tau\dot x_k)-b=E_k+\tau A\dot x_k=E_k+\tau r_k$. ∎

So explicit ZNN-Euler realises $E_{k+1}=E_k-h\Phi(E_k)$ with $h=\gamma\tau$, which is the **explicit Euler scheme of the error ODE**. Every non-Lipschitz activation then inherits the known pathologies of explicit Euler [PEB19, §1; JR20, Remark 9]:

* **Chattering.** A term $a|e|^p\operatorname{sign}e$ with $p<1$ settles into the period-2 orbit $e\mapsto -e$ with amplitude $e^\star=(ah/2)^{1/(1-p)}$. For the sign activation ($p=0$) this is $O(h)$; for sign-bi-power with $r=\tfrac12$ it is about $h^2/16$.
* **Blow-up for large errors.** A positive-degree term $a|e|^p$ with $p>1$ makes explicit Euler unstable when $h\,a|e_0|^{p-1}>2$ [PEB19, §1.2]. In the tests, explicit Euler diverges for the predefined-time activation of §5.
* **No finite-step convergence.**

The construction used throughout is therefore: **choose a consistent one-step map $\Psi_h$ of the scalar error ODE $\dot e=-\Phi(e)$ and request the rate**

$$r_k = \frac{\Psi_{h}(E_k)-E_k}{\tau},\qquad h=\gamma g(t_k)\tau, \tag{C3}$$

so that, by Lemma 1, $E_{k+1}=\Psi_h(E_k)$ on time-invariant problems. The time-derivative feed-forward $\partial_tE$ stays explicit, and only data at $t_k$ are used. The scheme is **causal**: $f$ is evaluated once per step, at $(t_k,x_k,k)$. This is exactly the idea of `ImplicitErrorEuler`, for which $\Psi_h$ is the resolvent; here it is generalised to maps that are consistent in the sense of [PEB19].

**Consistency [PEB19, Def. 2.1/2.2].** A discrete map is a *consistent* approximation of a finite-time stable system if three properties hold:

1. *Existence:* the implicit equation is solvable, and 0 is its unique solution from 0.
2. *Finite-time convergence:* every sequence reaches $0$ in finitely many steps. For fixed-time stable systems this is replaced by *practical fixed-time convergence*: $\|x_i\|\le r$ for $i\ge N(r)$, independently of $x_0$.
3. *Approximation:* $\|\varphi(h,x_i)-x_{i+1}\|\le h\,\omega(h)$ on compact sets that exclude the origin.

---

## 3. The four error maps $\Psi_h$ (element-wise; $h$ broadcast with shape $(B,1)$)

### 3.1 `resolvent`: backward Euler [AB10, HAB16]

$$\Psi^{\rm res}_h(e) = (I+h\Phi)^{-1}(e),\qquad\text{the unique } y \text{ with } y+h\Phi(y)\ni e .$$

This is `Activation.prox`, implemented by bisection with a dead zone $|e|\le h\Phi(0^+)$. It is consistent (exact zero in finitely many steps) **only when Φ has a jump at 0**. For a continuous non-Lipschitz Φ, backward Euler is merely asymptotically convergent. [PEB19, §1.1] shows this with $\dot x=-2\sqrt{|x|}\operatorname{sgn}x$, where $x_{i+1}=(\sqrt{h^2+|x_i|}-h)^2\operatorname{sgn}x_i\neq0$. For sign-bi-power with $r=\tfrac12$ the resolvent converges *quadratically*, because $y\approx(2e/h)^2$ near 0, but never reaches exactly 0. Test: `test_resolvent_is_not_finite_step_for_continuous_activation`.

### 3.2 `peb`: homogeneity-based consistent discretization [PEB19]

Numbers in parentheses in this subsection, such as (3.12) or (4.2), are [PEB19]'s own equation numbers.

**Ingredients [PEB19, §3].** A dilation is $\mathbf d(s)=e^{G_{\mathbf d}s}$, with $G_{\mathbf d}$ anti-Hurwitz. $P\succ0$ is chosen with $PG_{\mathbf d}+G_{\mathbf d}^\top P\succ0$ (3.2). The canonical homogeneous norm is

$$\|x\|_{\mathbf d}=e^{s_x}\quad\text{with}\quad \|\mathbf d(-s_x)x\|=1. \tag{3.4}$$

A field $f$ is $\mathbf d$-homogeneous of degree $\nu$ if $f(\mathbf d(s)x)=e^{\nu s}\mathbf d(s)f(x)$ (3.6). The change of coordinates

$$y=\Phi_{\mathbf d}(x)=\|x\|_{\mathbf d}\,\mathbf d(-\ln\|x\|_{\mathbf d})\,x,\qquad x=\mathbf d(\ln\|y\|)\,\frac{y}{\|y\|} \tag{3.12}$$

maps $\dot x=f(x)$ to

$$\dot y=\|y\|^{1+\nu}\tilde f(y),\qquad \tilde f(y)=\Big(\tfrac{(I-G_{\mathbf d})yy^\top P}{y^\top PG_{\mathbf d}y}+I\Big)f\big(\tfrac{y}{\|y\|}\big). \tag{3.9}$$

By Remark 3.10, the degree can be normalised to $\nu=\mp1$.

* **Theorem 4.1 ($\nu=-1$, finite time).** The implicit scheme $y_{i+1}\in y_i+h\tilde F(y_{i+1})$ (4.2), where $\tilde F$ is the Filippov regularisation of $\tilde f$, followed by $x_i=\Phi_{\mathbf d}^{-1}(y_i)$ (4.1), is a consistent approximation.
* **Theorem 5.1 ($\nu=+1$, practically fixed time).** The semi-implicit scheme $y_{i+1}=y_i+h\|y_i\|\,\|y_{i+1}\|\tilde f(y_{i+1})$ (5.3) is a consistent approximation, with $\|y_i\|\le 1/(ich)$, $c=\inf_{z\in S}-z^\top P\tilde f(z)$.

**Scalar specialisation (derived here; this is what the code does per component).** For one term $\dot e=-a|e|^p\operatorname{sign}e$ in $\mathbb R$ ($n=1$, $P=1$):

* $p<1$: the field is homogeneous of degree $p-1$ for $\mathbf d(s)=e^{s}$. Re-scaling with $G_{\mathbf d}=1/(1-p)$ gives $\nu=-1$. Then $\|e\|_{\mathbf d}=|e|^{1-p}$, and (3.12) gives $y=|e|^{1-p}\operatorname{sign}e$. In (3.9), $\tilde f(y)=\frac1{G_{\mathbf d}}f(\operatorname{sign}y)=-a(1-p)\operatorname{sign}y$. The inclusion (4.2) is then solved by the soft threshold:
  $$y_{+}=\operatorname{sign}(y)\max\{|y|-a(1-p)h,\,0\},\qquad e_{+}=\operatorname{sign}(y)\,|y_+|^{1/(1-p)}. \tag{C4}$$
  For $a=2$, $p=\tfrac12$ this is literally the map of [PEB19, §1.1]: $\hat x_{i+1}=(\sqrt{|\hat x_i|}-h)^2\operatorname{sgn}\hat x_i$ if $|\hat x_i|>h^2$, else 0. Test `test_peb_scalar_transcriptions`.
* $p>1$: $G_{\mathbf d}=1/(p-1)$ gives $\nu=+1$, $y=|e|^{p-1}\operatorname{sign}e$ and $\dot y=-a(p-1)|y|y$. Scheme (5.3) gives
  $$y_{+}=\frac{y}{1+a(p-1)h|y|},\qquad e_+=\operatorname{sign}(y)|y_+|^{1/(p-1)}. \tag{C5}$$
  For $a=1$, $p=2$ this is $x_{i+1}=x_i/(1+h|x_i|)$ [PEB19, §1.2].
* $p=1$ (degree 0, not covered by [PEB19]): ordinary implicit Euler $e_+=e/(1+ah)$.

Implementation note: (C5) is evaluated as $y_+=1/(1/y+a(p-1)h)$, which equals $y/(1+a(p-1)h|y|)$ and stays finite when $|e|^{p-1}$ overflows (§11, F4).

For a single term, (C4) and (C5) coincide with the **exact flow** of the scalar ODE: $\dot y$ is constant, respectively $y(t)=y_0/(1+a(p-1)|y_0|t)$.

**Sums of homogeneous terms: Lie splitting (ASSUMPTION / novel extension).** Sign-bi-power $\Phi(e)=\tfrac12(\operatorname{sig}^r e+\operatorname{sig}^{1/r}e)$ [LCL13], and the fixed-time field $\alpha|e|^p+\beta|e|^q$ with $p<1<q$ [Pol12], are not homogeneous. They are homogeneous "in the bi-limit", which [PEB19] does not cover. Their §6.1 instead *switches* between the negative- and positive-degree laws. We compose the single-term PEB maps in order of **decreasing degree**, so that the finite-time dead zone is applied last:

$$\Psi^{\rm peb}_h=\Psi^{(p_1)}_{a_1h}\circ\cdots\circ\Psi^{(p_m)}_{a_mh},\qquad p_1<\dots<p_m .$$

**Proposition 1 (discrete fixed-time bound for the split map).** Let $\Phi=a_1\operatorname{sig}^{p_1}+a_2\operatorname{sig}^{p_2}$ with $p_1<1<p_2$. Every sequence $e_{k+1}=\Psi^{\rm peb}_h(e_k)$ is sign-preserving, non-increasing in $|e|$, and reaches **exactly 0** after at most

$$N\le\Big\lceil\frac1{a_2(p_2-1)h}\Big\rceil+\Big\lceil\frac1{a_1(1-p_1)h}\Big\rceil$$

steps, **for every** $e_0$. This mirrors the continuous bound $T\le\frac1{a_1(1-p_1)}+\frac1{a_2(p_2-1)}$ of [Pol12] / [AL19, Eq. (2)], and [PEB19, §6.1]'s $N\le2/h$.

*Proof.*

1. Write $s=|e|$, with $S_2$ the positive-degree map (C5) and $S_1$ the finite-time map (C4). Both are monotone non-decreasing and satisfy $S_i(s)\le s$.
2. From $s_{k+1}=S_1(S_2(s_k))\le S_2(s_k)$, the comparison lemma [JR20, Lemma 13] gives $s_k\le S_2^k(s_0)$.
3. In the coordinate $y=s^{p_2-1}$, $S_2^k$ gives $y_k=y_0/(1+kcy_0)\le1/(kc)$ with $c=a_2(p_2-1)h$. Hence $s\le1$ after $k_1=\lceil1/c\rceil$ steps.
4. From then on $s_{k+1}\le S_1(s_k)$, so $s^{1-p_1}$ decreases by at least $a_1(1-p_1)h$ per step until it reaches 0. ∎

Test: `test_peb_split_discrete_fixed_time_bound`, with $r\in\{0.3,0.5,0.8\}$ and $e_0$ up to $10^{12}$. The split map is first-order accurate, which satisfies the approximation property (Def. 2.1).

### 3.3 `exact`: exact flow in the settling-time coordinate (Jiménez-Rodríguez et al. 2020 and generalisation)

**[JR20], verbatim.** The predefined-time class (1) is

$$\dot x=-\frac{1}{\rho_1(1-\rho_2)}\frac{\kappa(|x|)^{\rho_2}}{\kappa'(|x|)}\operatorname{sign}x,$$

with $\rho_1>0$, $0\le\rho_2<1$ and $\kappa\in\mathcal K^1$: strictly increasing, $\kappa(0)=0$, $\kappa(\infty)=1$. Its settling time is $T(x_0)=\rho_1\kappa(|x_0|)^{1-\rho_2}<\rho_1$. The transformation (4) $w=y_{\rho_2}(x)=\kappa(|x|)^{1-\rho_2}\operatorname{sign}x$ gives $\dot w=-\rho_1^{-1}\operatorname{sign}w$ (8), and implicit Euler on (8) yields the **exact** discretization (7):

$$x_{k+1}=\kappa^{-1}\Big(\max\big\{\kappa(|x_k|)^{1-\rho_2}-h/\rho_1,\;0\big\}^{\frac1{1-\rho_2}}\Big)\operatorname{sign}x_k. \tag{JR-7}$$

The results used are Theorem 6 ($x_k=x(kh)$) and Corollary 7 ($x_k=0$ for $k\ge\lceil\rho_1/h\rceil$). With ZNN gain γ, $h=\gamma\tau$, so the discrete predefined time is $\lceil\rho_1/(\gamma\tau)\rceil$ steps. Implemented as `PredefinedTimeJR.exact_step`. The perturbed version (13) is not used, see §7.

**Generalisation (derived here).** For *any* odd increasing finite-time activation, the time-to-go $T(s)=\int_0^s\mathrm dz/\Phi(z)$ defines the coordinate $w=T(|e|)\operatorname{sign}e$. Along $\dot e=-\Phi(e)$ it satisfies $\dot w=-\operatorname{sign}w$, whose implicit Euler step is exact. Hence

$$\Psi^{\rm exact}_h(e)=\operatorname{sign}(e)\,T^{-1}\big(\max\{T(|e|)-h,0\}\big). \tag{C6}$$

[JR20]'s $\rho_1\kappa^{1-\rho_2}$ is precisely $T$. All exact maps are clipped to $|e_{k+1}|\le|e_k|$ (the exact flow is non-increasing in $|e|$); the clip is inactive except when $h$ is below the rounding error of $T(|e|)$ (§11, F3). The construction coincides with [PEB19]'s Fig. 1.1 scheme, $x_{i+1}=\Phi^{-1}(\Phi(x_i)+h\tilde f(\Phi(x_{i+1})))$, with $\Phi=T$. Closed forms implemented:

* **Single power** $a|e|^p$: (C4) for $p<1$, $e\,e^{-ah}$ for $p=1$, (C5) for $p>1$.
* **Bernoulli, $(a_1,p_1<1)+(a_2,1)$:** with $w=|e|^{1-p_1}$, $\dot w=-(1-p_1)(a_1+a_2w)$, so
  $$w_+=\max\{w\,e^{-a_2(1-p_1)h}+\tfrac{a_1}{a_2}\operatorname{expm1}(-a_2(1-p_1)h),0\}.$$
* **Bernoulli, $(a_1,1)+(a_2,p_2>1)$:** with $v=|e|^{p_2-1}$, $\phi=v/(1+\tfrac{a_2}{a_1}v)$ decays as $e^{-a_1(p_2-1)t}$.
* **Bi-power $(a_1,p_1)+(a_2,p_2)$ with $p_1<1<p_2$** (sign-bi-power, Polyakov fixed-time field). Substituting $U=\frac{a_2}{a_1}z^{q}$ with $q=p_2-p_1$ gives
  $$T(s)=\frac{(a_1/a_2)^{\beta}}{a_1q}\int_0^{U(s)}\frac{u^{\beta-1}}{1+u}\,\mathrm du=C\,I_{\frac{U}{1+U}}(\beta,1-\beta),\qquad \beta=\frac{1-p_1}{q}\in(0,1),\quad C=\frac{(a_1/a_2)^{\beta}\,\pi}{a_1q\sin\pi\beta}, \tag{C7}$$
  where $I_z$ is the regularised incomplete beta function, inverted with `scipy.special.betaincinv`. Its supremum $C=\Gamma(m_p)\Gamma(m_q)\,(a_1/a_2)^{m_p}/(a_1q)$, with $m_p=\beta$ and $m_q=1-\beta$, is **exactly the least upper bound of [AL19, Thm 1]** ($k=1$). Formula (C7) extends it to the full settling-time function $T(e_0)$. For sign-bi-power with $r=\tfrac12$, $C=4.8368/\gamma$. Precision near $z\to1$ is preserved by evaluating the complementary branch $I_{1/(1+U)}(1-\beta,\beta)$.

**Proposition 2 (discrete fixed time with the sharp bound).** For the bi-power field, $\Psi^{\rm exact}_h$ reaches 0 after exactly $\lceil T(e_0)/h\rceil$ steps, which is at most $\lceil C/h\rceil$ for every $e_0$.

Tests:

* `test_exact_map_is_exact_flow`: agreement with DOP853 at rtol $10^{-13}$ to within $10^{-12}$ for 7 activations.
* `test_bipower_settling_time_formula`: [AL19] Thm 1 against quadrature, and step counts.
* `test_jr_exact_discretization`: three κ families × $\rho_2\in\{0,0.5\}$; checks Theorem 6 and Corollary 7, with $e_0$ up to $10^8$.

### 3.4 `semi_implicit`: frozen-coefficient (linearly implicit) Euler

Write $\Phi(e)=j\operatorname{sign}e+\kappa(e)e$, with $j=\Phi(0^+)$ and $\kappa(e)=(\Phi(e)-j\operatorname{sign}e)/e\ge0$. Then

$$e'=\frac{e}{1+h\kappa(e_k)},\qquad e_+=\operatorname{sign}(e')\max\{|e'|-hj,0\}. \tag{C8}$$

The first step is the semi-implicit idea of [PEB19, §1.2 and (5.3)], applied in the original coordinate. The second steps the jump exactly (soft threshold) [AB10]. The map needs one activation evaluation and no root solve, and it works for **every** activation, including `power_sigmoid`.

Properties:

* It is sign-preserving and non-expansive for all $h>0$.
* It is practically fixed-time for positive-degree terms. For example, $\kappa=|e|^{p-1}$ gives $|e_k|^{-(p-1)}\ge k(p-1)h$.
* It is finite-step exact when Φ has a jump.
* For continuous finite-time Φ it converges super-linearly but not exactly. This is **not** consistent in the sense of Def. 2.1, and is flagged as such.

**Not implemented, but documented:** the Rosenbrock linearisation $(1+h\Phi'(e_k))(e_{k+1}-e_k)=-h\Phi(e_k)$. For $\Phi=|e|^p\operatorname{sign}e$ it gives
$$e_{k+1}=e_k\,\frac{1-h(1-p)|e_k|^{p-1}}{1+hp|e_k|^{p-1}}\;\to\;-\frac{1-p}{p}\,e_k \quad(e_k\to0).$$
This oscillates, and is locally *unstable* for $p<\tfrac12$. It is therefore inconsistent, and the frozen-coefficient form is used instead.

---

## 4. Higher-order look-ahead with error memory (`implicit_taylor_zhang`), NOVEL

### 4.1 Why the naive combination fails

The Taylor–Zhang formula [TZ15] is $c=(1,-\tfrac32,1,-\tfrac12)$, i.e. $\tau\dot x_k=x_{k+1}-\tfrac32x_k+x_{k-1}-\tfrac12x_{k-2}+O(\tau^3)$. Its conditions are $\sum c=0$, $\sum c_i(-i)=1$ and $\sum c_ii^2=0$. Zero-stability: the roots of the characteristic polynomial are $1$ and $0.25\pm0.661i$, with modulus $0.707$. Plugging the resolvent rate $r_k=(\Psi_h(E_k)-E_k)/\tau$ into (C2) gives, on constant data,

$$E_{k+1}=\Psi_h(E_k)+\tfrac12\big(E_k-2E_{k-1}+E_{k-2}\big).$$

Inside the dead zone ($\Psi_h=0$) the error obeys $E_{k+1}=\tfrac12\Delta^2E_k$, with characteristic polynomial $(z-\tfrac12)(z^2+1)$. That is **marginally stable** (roots $\pm i$): the memory term keeps re-exciting the error and finite-step convergence is lost. For linear Φ the error recursion $z^3-(\tfrac1{1+h}+\tfrac12)z^2+z-\tfrac12$ is stable but tends to marginal as $h\to\infty$:

| $h$ | 0.3 | 1 | 1.5 | 5 | 50 |
|---|---|---|---|---|---|
| explicit TZ, $\rho$ of $z^3-(\tfrac32-h)z^2+z-\tfrac12$ | 0.815 | **1.000** | **1.086** | **3.13** | **48.5** |
| naive implicit TZ, $\rho$ | 0.800 | 0.879 | 0.907 | 0.965 | 0.996 |
| error-memory TZ, $\rho=1/(1+h)$ | 0.769 | 0.500 | 0.400 | 0.167 | 0.020 |

### 4.2 The error-memory rate

Request instead

$$\tau r_k=c_{-1}\Psi_h(E_k)+\sum_{j=0}^{s-1}c_jE_{k-j}\qquad\text{(EM)}$$

and keep the update (C2) unchanged. The past errors $E_{k-j}=E(x_{k-j},d_{k-j})$ were computed at earlier steps from data available then, so the scheme stays causal. They are recorded by the request object.

**Lemma 2 (exact error map for any formula).** For $E=Ax-b$ with constant $A$ and $b$, (EM) combined with (C2) gives $E_{k+1}=\Psi_h(E_k)$ **exactly**, whatever the coefficients $c$ (with $\sum c=0$).

*Proof.* Since $\sum_{j\ge0}c_j=-c_{-1}$,
$$c_{-1}E_{k+1}=A(\tau\dot x_k-\textstyle\sum c_jx_{k-j})-c_{-1}b=\tau r_k-\textstyle\sum c_j(E_{k-j}+b)-c_{-1}b=c_{-1}\Psi_h(E_k).$$ ∎

Test `test_time_invariant_error_map_reduction_with_memory` checks $r_{k+1}/r_k=1/(1+h)$ to $10^{-8}$ in the multistep phase. The naive variant fails this check.

**Interpretation.** For Taylor–Zhang, (EM) is equivalent to

$$x_{k+1}=x_k+J_k^{-1}\big(\Psi_h(E_k)-E_k\big)-\tau J_k^{-1}\partial_tE_k+\tfrac12\,\Delta^2\hat x_k,\qquad \hat x_{k-j}=x_{k-j}-J_k^{-1}E_{k-j}.$$

This combines a Newton/ZNN correction with a consistent error map, plus the Taylor–Zhang curvature term applied to **Newton-projected past states**. For a linear problem, $\hat x_{k-j}=x^*_{k-j}+(I-A_k^{-1}A_{k-j})(x_{k-j}-x^*_{k-j})=x^*_{k-j}+O(j\tau)\delta_{k-j}$.

**Proposition 3 (residual order, sketch).** Let $\delta_k=x_k-x^*_k$. On a smooth time-varying problem, using the formula's consistency on $x^*$:

$$\delta_{k+1}=J_k^{-1}\Psi_h(E_k)+\textstyle\sum_j c_j(J_k^{-1}J_{k-j}-I)\delta_{k-j}/c_{-1}+O(\tau^{q+1}),$$

where $q+1$ is the formula's order of $\tau\dot x$ accuracy. The memory enters only through $O(\tau)$ couplings. Hence:

* For non-expansive, contracting $\Psi_h$ the steady residual is $O(\tau^{q+1})$ at fixed $h$, the same order as the explicit formula.
* For a map with a dead zone the residual equals **one local truncation error**, with no $1/h$ amplification. Test `test_sign_type_residual_is_one_step_truncation_error`: the linear/sign floor ratio is $(1+h)/h$.
* The error stability is that of $\Psi_h$, i.e. unconditional in $h$, rather than the formula's $h$-window. Explicit TZ is unstable for $h>1$.

**ASSUMPTION / not proven here:** the $O(\tau)$ couplings must be small, i.e. $\tau\sum_j|c_j|\,j\,\|J^{-1}\dot J\|\lesssim1$. A rigorous theorem with constants is left open.

### 4.3 Generality

Lemma 2 holds for any look-ahead formula, so `consistent(formula=<any registered LookAhead key>, error_map=...)` works with the ZeaD formulas of the sibling `zead` area. Measured with $h=0.1$, a sign activation and the exact map, the residual slope equals the formula's expected order for all 17 registered LookAhead formulas:

* 2 for Euler-type formulas;
* 3 for TZ and the 4i formulas;
* 4 for 5i, 6i and `uhlig_2_3`;
* 5 for 7i, 8i and `uhlig_3_4`;
* 6 for 10i, 11i and `uhlig_4_6`.

Constant-data runs reach $\le10^{-15}$. `uhlig_3_4` diverges explicitly at $h=0.1$ but is stable with (EM). This is an observation: by Proposition 3 the parasitic roots are multiplied by $O(\tau)$. Test `test_error_memory_with_registered_zead_formula` uses `zead_6i_guo2017` and skips if that key is absent.

### 4.4 Relation to prior work

Prediction-correction methods [SMK16] also separate a *prediction* (Taylor extrapolation of the solution trajectory) from a *correction* (Newton or gradient steps on the current problem). (EM) can be read as a ZNN analogue with three differences:

* the correction is a consistent finite-/fixed-time error map rather than a fixed number of Newton steps;
* the prediction uses only past *projected* samples, in the multistep (ZeaD) form;
* the step is a single causal model evaluation per sample.

To our knowledge the combination is not in [PEB19], [JR20], or the ZeaD literature. **Claim of novelty to be checked by the authors.**

---

## 5. Activations provided as test vehicles

The two classes below are local test vehicles (resolved by key locally, see the header; not in the shared registry). Sign, sign-bi-power, SFAF, WSBP, VAF and the fixed-time bi-power activation themselves live in `activations.py`; their power-sum structure is recognised automatically (§6).

* `PowerSum(terms=[(a_i, p_i)])` (key `signed_power_sum`): $\Phi(e)=\sum_ia_i\operatorname{sign}(e)|e|^{p_i}$ with $a_i>0$, $p_i\ge0$. It covers:
  * sign, with $p=0$ [ZDQ17, AB10];
  * powers;
  * sign-bi-power, with terms $(\tfrac12,r),(\tfrac12,1/r)$ [LCL13];
  * the bi-power fixed-time field [Pol12; AL19 Eq. (1), $k=1$];
  * the strongly predefined-time field $\frac{\gamma_{AL}}{T_c}(\alpha|e|^p+\beta|e|^q)$ [AL19 Eq. (3), $k=1$], by scaling the $a_i$. Its least upper settling bound is $T_c$; the case $k\ne1$ is **not** covered.

  It also exposes `power_terms()`, the duck-typed hook used by the `peb` and `exact` maps. Siblings may implement this hook on their own activations. `uniform_bound(gamma)` returns the smallest [AL19] Thm 1 value over term pairs $p_i<1<p_j$ (dropping positive terms only lengthens the settling time; exact least upper bound for two terms), and $\infty$ without such a pair. `convergence` is `finite`, `fixed`, `exponential` or `asymptotic` according to the exponents.
* `PredefinedTimeJR(kappa, a, mu, rho1, rho2)`: [JR20] Eq. (1) with three κ families:
  * `arctan`: $\kappa=\frac2\pi\arctan(ar^\mu)$. [JR20, Ex. 4] for $\mu=1$. For $\rho_2=0$ it gives $\Phi=\frac{\pi}{2a\mu\rho_1}(|e|^{1-\mu}+a^2|e|^{1+\mu})\operatorname{sign}e$, the symmetric bi-power predefined-time activation.
  * `exp`: $\kappa=1-e^{-ar^\mu}$. **Reparametrised:** [JR20, Ex. 4] writes $1-a_{JR}^{-r}$ with $a_{JR}>1$; the code's $a$ is $\ln a_{JR}>0$ ($\mu=1$). For $\rho_2=0$ it gives the [ST18] class $\frac{1}{a\mu\rho_1}e^{a|e|^\mu}|e|^{1-\mu}\operatorname{sign}e$, which is the zoo activation `predefined_exp` (Li, Liao, Xiao, Lu 2019, F2) with $k_2=k_3=0$, $a=1$, $\mu=p$, $\rho_1=1/k_1$. The `exact` map uses (JR-7) for that case automatically.
  * `rational`: $\kappa=r^\mu/(r^\mu+a)$. [JR20, Ex. 4] for $\mu=1$.

  `settling_time(e0, gamma)` $=\rho_1\kappa(\max_i|e_{0,i}|)^{1-\rho_2}/\gamma$ (exact, JR §2.2), `upper_bound_time` = `uniform_bound` $=\rho_1/\gamma$, `convergence = "predefined"`.

  **ASSUMPTION:** the compositions $\kappa(r^\mu)$ with $0<\mu<1$ are still in $\mathcal K^1$, which is what [JR20]'s results require. They are not listed in [JR20]. $\mu>1$ is rejected because it would make Φ unbounded at 0.

---

## 6. Interface and protocol

* **Request object.** The discretizer passes an `ErrorMapRequest` as the `implicit` argument of `f`. The object is truthy and carries the map name and the formula coefficients, and records $E_{k-j}$. A model that honours it calls `req.desired_Edot(act, ctx, g)`, with `g = gamma*gain(t)`, and uses the returned rate in $J\dot x=r-\partial_tE$.
* **Opt-in model.** `ConsistentZNN` (key `znn_consistent`) honours the request. Without one it is bit-identical to `DesignFormulaZNN` (test `test_consistent_model_is_plain_znn_without_request`).
* **Fallback.** Any other implicit-aware model, such as the base `znn` or `znn_explicit_inverse`, sees a truthy `implicit` and uses its **resolvent** rate. The discretizer detects that the request was not honoured and warns once. Euler variants then reproduce `implicit_euler_err` exactly (test `test_base_model_fallback_warns_and_matches_resolvent`). TZ variants then become the naive ablation.
* **Core change requested** (`docs/spec/CORE_CHANGE_REQUESTS.md`, section `consistent`): three lines in `DesignFormulaZNN.desired_Edot` would make the base `znn` model, and every subclass, honour the protocol.
* **Structure detection.** `peb` and `exact` need $\Phi=\sum_ia_i\operatorname{sig}^{p_i}$ or an `exact_step`. Besides the `power_terms()`/`exact_step()` hooks, the zoo activations `linear`, `sign`, `sign_power`, `sign_bi_power`, `sfaf`, `wsbp`, `vaf`, `fixed_time`, `power` (odd integer $p$) and `power_sum` (Yang–Zhang) are decomposed from their own parameters. Every decomposition is checked against the activation at 9 probe points (rtol $10^{-12}$); on a mismatch the decomposition is discarded with a warning, so a sibling's formula change cannot silently corrupt the maps. Closed-form exact maps then exist for `sign`, `sign_power`, `sign_bi_power`, `sfaf` (Bernoulli), `power`, `linear`, and `predefined_exp` with $k_2=k_3=0$; the 3–4-term activations (`wsbp`, `vaf`, `fixed_time` with $k_3>0$, `power_sum`) use the Lie-split `peb` map.
* **Map fallback.** `exact` falls back to `peb`, and `peb` to `exact`, then both to `resolvent` when no closed form exists, with a warning once. For example, `power_sigmoid` gives `resolvent`, and `jr_predefined` with `peb` gives `exact`.
* **Defaults.** No hidden parameters: everything follows from the activation's own parameters and $h=\gamma g(t_k)\tau$. Start-up: Euler (one-step rate) until the formula's history is full. The auxiliary state $z$ is advanced with forward Euler.

**Cost per call** (sign-bi-power, $B=4$, Apple M-series, numpy):

| $m$ per trajectory | explicit Φ | `peb` | `semi_implicit` | `exact` (betaincinv) | `resolvent` (80-step bisection) |
|---|---|---|---|---|---|
| 16 | 2.5 µs | 7.3 µs | 10 µs | 93 µs | 444 µs |
| 1024 | 18 µs | 32 µs | 36 µs | 4.8 ms | 2.1 ms |

`peb` and `semi_implicit` are the cheap options, at about 2× an explicit activation evaluation.

---

## 7. Assumptions and limitations (flagged)

1. [PEB19] equation numbers refer to HAL preprint v1. [HAB16] and [ST18] were not read in full.
2. **Lie splitting** of non-homogeneous sums (§3.2) is our extension. [PEB19] treats homogeneous fields, and switching between degrees in §6.1.
3. The **closed-form settling-time maps** (C6)–(C7) and the Bernoulli maps are our derivations, verified numerically against DOP853 to $10^{-12}$. Only the JR class map is verbatim [JR20, Eq. (7)].
4. **Error memory (EM)** and Propositions 1–3 are new. Proposition 3 is a sketch; its constants are not derived.
5. With a time-varying gain $g(t)$, $h$ is frozen at $t_k$. Exactness (Theorem 6) holds for constant gain only.
6. The perturbed [JR20] scheme (13), $\Psi(E_k+hf_k)$, is not implemented, because the ZNN perturbation (time variation, noise) is unknown at $t_k$. In `simulate`, noise enters after the map: $E_{k+1}=\Psi_h(E_k)+\tau\nu_k$. A dead zone wider than $\tau|\nu|$ then rejects it in one step, but the theorem of [JR20, Prop. 10] does not apply verbatim.
7. Lemma 1 and Lemma 2 are exact for problems that are affine in $x$ with constant data. For nonlinear $E$ the error map holds up to the Newton remainder $O(\|x_{k+1}-x_k\|^2)$.
8. **Zoo decompositions** (§6) are read from the sibling activations' parameter names; the probe check guards against drift. The `predefined_exp` ↔ (JR-7) equivalence is derived here (DERIVED), and checked against DOP853 in the tests.
9. `chattering` (from `metrics.py`) measures differences of $\|E\|$. A symmetric period-2 cycle on a *time-invariant* problem has constant norm and scores 0, so the chattering comparisons are run on the time-varying problem. On the time-invariant problem the explicit $O(h)$ floor is measured by MSSRE instead.

---

## 8. Tests (`tests/test_consistent.py`)

**Structure**

* `test_registry_and_causality`: every key builds, is causal, and calls `f` once per step at $(t_k,x_k,k)$.
* `test_taylor_zhang_coefficients_consistent_and_zero_stable`: order conditions, zero-stability, and the marginal dead-zone polynomial $(z-\tfrac12)(z^2+1)$.

**Maps**

* `test_exact_map_is_exact_flow`
* `test_jr_exact_discretization`: JR Theorem 6 and Corollary 7.
* `test_bipower_settling_time_formula`: [AL19] Theorem 1.
* `test_peb_scalar_transcriptions`: [PEB19] §1.1 and §1.2.
* `test_peb_split_discrete_fixed_time_bound`: Proposition 1.
* `test_resolvent_is_not_finite_step_for_continuous_activation`
* `test_maps_sign_preserving_and_nonexpansive`: 4 maps × 6 activations, $h$ up to 100.

**(a) Time-invariant problem** (`linear_system` with `freq=0`, so $\partial_tE\equiv0$)

* `test_time_invariant_finite_step_convergence`: 6 schemes × {sign, sign-bi-power, JR}. The residual goes below $10^{-13}$ in fewer than 150 steps and stays there.
* `test_time_invariant_explicit_sign_chatters_at_order_h`: $0.5<\text{MSSRE}/h<3$ for $h\in\{0.1,0.2,0.4\}$, while the exact map gives below $10^{-13}$.
* `test_time_invariant_error_map_reduction_with_memory`: Lemma 2.
* `test_predefined_time_in_discrete_time`: residual about 0 for $k\ge\lceil\rho_1/(\gamma\tau)\rceil$ with $\|x_0\|\sim10^3$, while explicit Euler diverges.

**(b) Time-varying `linear_system`**

* `test_tv_residual_order`: 7 schemes × {linear, sign}. Slope 2 for Euler and 3 for TZ, ±0.3, at $h=0.3$.
* `test_sign_type_residual_is_one_step_truncation_error`
* `test_error_memory_tz_unconditionally_stable`: explicit TZ diverges at $h=5$; EM-TZ keeps slope 3 at $h=5$ and $h=50$.
* `test_error_memory_with_registered_zead_formula`

**(c) Chattering**

* `test_chattering_reduced_vs_explicit`: sign-bi-power and sign, at $h=0.3$. The consistent chattering index is below half the explicit one and below 0.05, and the MSSRE is below 0.2× the explicit one. In the ablation, the naive TZ chatters more than 5× the EM-TZ.

**Interface**

* `test_base_model_fallback_warns_and_matches_resolvent`
* `test_consistent_model_is_plain_znn_without_request`
* `test_exact_falls_back_for_activation_without_closed_form`

**Added in verification** (§11)

* `test_jr_example_8`: [JR20] Example 8, Thm 6 against the closed-form solution (2) coded independently, Cor. 7, and the explicit-Euler period-2 orbit at the paper's $h=0.0651318636497$.
* `test_activation_keys_and_contract`: local keys, no injection into the shared registry, activations.py attribute contract, [AL19] Thm 1 `uniform_bound`, JR `settling_time`/`uniform_bound`.
* `test_scheme_names_match_registry_keys`.
* `test_zoo_activation_power_terms` (10 zoo keys): decomposition reproduces the activation; `peb` no longer falls back.
* `test_zoo_exact_map_matches_zoo_settling_time`: step counts equal $\lceil T(e_0)/h\rceil$ with $T$ the closed forms stated in `activations.py` (sign, sign_power, sfaf).
* `test_power_terms_guard_rejects_mismatch`, `test_predefined_exp_exact_map`.
* `test_batch_per_trajectory_gain`: per-trajectory $\gamma$ (shape $(B,1)$) gives the same traces as separate runs.
* `test_maps_finite_for_huge_errors`; `test_maps_sign_preserving_and_nonexpansive` extended to $|e|\in[10^{-10},10^{10}]$, $h=10^{-18}$, the JR `exp` family with $\rho_2=0.5$ and a Bernoulli-high field.

---

## 9. Reference numbers

All runs use `linear_system` with n=4, κ=10, seed=1, T=4 and τ=1e-2.

**Time-varying problem, $h=\gamma\tau=0.3$** (MSSRE / chattering index):

| activation | explicit Euler | explicit TZ | `implicit_euler_err` | `peb_euler` = `exact_flow_euler` | `semi_implicit_euler` | `implicit_taylor_zhang` | `peb_taylor_zhang` | TZ naive |
|---|---|---|---|---|---|---|---|---|
| sign-bi-power $r=\frac12$ | 1.2e-2 / 0.043 | 5.0e-2 / 0.084 | 8.5e-4 / 0.011 | 8.3e-4 / 0.011 | 9.8e-4 / 0.012 | 3.8e-5 / 0.014 | 3.8e-5 / 0.014 | 2.8e-3 / **0.60** |
| sign | 0.39 / 0.11 | 0.71 / 0.12 | 8.3e-4 / 0.011 | 8.3e-4 / 0.011 | 8.3e-4 / 0.011 | 3.8e-5 / 0.014 | 3.8e-5 / 0.014 | 0.54 / 0.20 |
| `jr_predefined` (arctan, ρ1=1) | **diverges** | **diverges** | 8.3e-4 / 0.011 | 8.3e-4 / 0.011 | 8.3e-4 / 0.011 | 3.8e-5 / 0.014 | 3.8e-5 / 0.014 | 0.76 / 0.80 |

**Empirical orders at $h=0.3$**, with τ ∈ {3e-2, 1e-2, 3e-3, 1e-3}:

* linear Φ: all Euler-based schemes 1.99–2.00, all TZ-based schemes 2.99;
* sign-type Φ: 2.00 and 3.00;
* naive TZ with sign-type Φ: 0.02–1.1 (order lost).

**Large gain, linear Φ, EM-TZ slope / MSSRE at τ=1e-3:**

| $h$ | slope | MSSRE |
|---|---|---|
| 0.9 | 3.00 | 8.0e-8 |
| 1.5 | 3.00 | 6.3e-8 |
| 5 | 3.00 | 4.6e-8 |
| 50 | 3.00 | 3.9e-8 |

Explicit TZ diverges for $h\ge1.5$. The naive variant at τ=3e-2 reaches MSSRE 0.36 at $h=5$ and 18.0 at $h=50$.

---

## 10. Open issues

* State Proposition 3 with constants. Prove existence and uniqueness for the vector, non-separable PEB scheme if norm-based activations are added to the zoo.
* Merge the `CORE_CHANGE_REQUESTS` item so that the base `znn` key benefits.
* Encourage sibling activation authors to expose `power_terms()` and/or `exact_step()`.
* Settling-time maps for $(\alpha|e|^p+\beta|e|^q)^k$ with $k\neq1$ [AL19] need numerical inversion and are not implemented.

---

## 11. Verification (adversarial review, 2026-10-18)

Every registered component was re-derived and re-checked against the primary sources and with independent numerics (scratch scripts outside the repository plus the tests listed in §8).

### 11.1 Sources re-read (full text) and equations compared symbol by symbol

| item | source checked | verdict |
|---|---|---|
| (C4), PEB §1.1 map $\hat x_{i+1}=(\sqrt{|\hat x_i|}-h)^2\operatorname{sgn}\hat x_i$ for $|\hat x_i|>h^2$, else 0 | [PEB19] HAL v1 full text, §1.1 | correct |
| (C5), PEB §1.2 semi-implicit $x_{i+1}=x_i/(1+h|x_i|)$ | [PEB19] §1.2 | correct |
| Scalar specialisation of (3.12), (3.9), (4.2), (5.3): $\|e\|_{\mathbf d}=|e|^{1-p}$ ($|e|^{p-1}$), $y=|e|^{1-p}\operatorname{sign}e$, $\tilde f=-a(1-p)\operatorname{sign}y$ (resp. $-a(p-1)\operatorname{sign}y$, giving $y_+=y/(1+a(p-1)h|y|)$) | [PEB19] §3–5, Thms 4.1 and 5.1, Remark 3.10, re-derived by hand | correct |
| Def. 2.1/2.2 (existence, finite-time / practical fixed-time convergence, approximation) and §6.1 $N\le2/h$ | [PEB19] §2 and §6.1 | correct |
| (JR-7), Eq. (1), (2), (4), (8), Thm 6, Cor. 7, Lemma 13, Example 4 κ-list | [JR20] arXiv:2001.08318v1 full text | correct, except that the `exp` family is reparametrised ($a=\ln a_{JR}$); now stated in the docstring and §5 |
| $C=(a_1/a_2)^\beta\pi/(a_1q\sin\pi\beta)$ = [AL19] Thm 1, Eq. (6) with $k=1$; [AL19] Eq. (2) bound used in Prop. 1 | [AL19] arXiv:1809.07012 full text; $\Gamma(\beta)\Gamma(1-\beta)=\pi/\sin\pi\beta$ | correct |
| Taylor–Zhang $c=(1,-\tfrac32,1,-\tfrac12)$ | order conditions $\sum c=0$, $\sum c_i(-i)=1$, $\sum c_ii^2=0$ and equality with `multistep.taylor_zhang` (now a test). The [TZ15] full text was not accessible (publisher 403) | correct by order conditions; primary text not re-read |
| All 11 DOIs of §1 | Crossref API (title, authors, volume, pages) | correct |

### 11.2 Independent numerical checks

* Bi-power exact map: $T(\Psi_h(e))=\max\{T(e)-h,0\}$ with $T$ by adaptive quadrature, for 5 coefficient sets (including $p_1=0$, asymmetric $a_1\ne a_2$, $p_2=5$), $e_0\in[10^{-9},10^8]$, $h\in[10^{-6},0.7]$: worst relative error $1.1\times10^{-15}$. $C$ against quadrature: $4.8367983$ for sign-bi-power $r=\tfrac12$.
* Spectral radii of §4.1 table recomputed (explicit TZ 0.815/1.000/1.086/3.13/48.5; naive 0.800/0.879/0.907/0.965/0.996): correct. TZ parasitic roots $0.25\pm0.661i$, modulus $1/\sqrt2$: correct.
* Lemma 2 and the TZ "Newton-projected" identity re-derived by hand: correct.
* Error memory with all 18 registered `LookAhead` formulas (seed 2, sign activation, exact map, $h=0.1$): residual slope equals `expected_order` within 0.1, except `zead_10i_wu2020` (6.43 for order 6; its floor at τ = 3e-3 is near round-off), constant-data residual $\le10^{-15}$. Explicit `uhlig_3_4` at $h=0.1$ has error-recursion spectral radius 1.045 (unstable), confirming §4.3.
* §9 reference numbers re-measured for sign-bi-power at $h=0.3$: reproduced (e.g. explicit Euler 1.22e-2/0.043, `peb_taylor_zhang` 3.78e-5/0.0144).
* Smoke run of `euler`, `implicit_euler_err`, `peb_euler`, `peb_taylor_zhang`, `taylor_zhang`, `exact_flow_taylor_zhang` with `znn_consistent`+sign-bi-power on all 13 registered problems: no crash; consistent schemes reduce MSSRE by 1–3 decades except on `tvqp_ineq` (nonsmooth complementarity residual), where EM-TZ has MSSRE comparable to explicit TZ and chattering index 0.56 (Euler-based consistent schemes are fine there). **Limitation:** Lemma 2/Prop. 3 assume a smooth affine-in-$x$ error.
* Info regimes `bd1`/`bd2`/`none`, design-space noise and `data_noise` run without error; with constant noise $\nu$ the consistent schemes settle at the one-step floor $\tau\|\nu\|$.
* Per-call cost re-measured (m = 16, B = 4): `peb` 7.2 µs, `semi_implicit` 10.6 µs, `exact` 96 µs, `resolvent` 463 µs; unchanged by the fixes.

### 11.3 Findings and fixes

* **F1 (activation key collision).** `PowerSum` used the key `power_sum`, which `activations.py` already assigns to the Yang–Zhang activation $\sum_k e^{2k-1}$; the guarded registration therefore never happened, contrary to the report, and model names `znn_power_sum` were ambiguous. Renamed to `signed_power_sum`.
* **F2 (cross-area test coupling).** Injecting `jr_predefined` into the shared `activations.REGISTRY` with the base-class defaults `convergence="exponential"`, `source=""` (while `finite_time=True`) violated the zoo contract checked by `tests/test_activations.py::test_registry_contains_zoo` in a full-suite run. With the opposite collection order, `test_formula_matches_source` and `test_local_exponential_rate_is_slope0` also failed (KeyError / wrong slope0) for the injected keys. Fix: the classes now carry `convergence`, `source`, `provenance`, contract-conforming `settling_time(e0, gamma)`, `upper_bound_time`, `uniform_bound`, and they are resolved locally (`get_activation`, model `znn_consistent`) instead of being injected. `pytest tests/test_consistent.py tests/test_activations.py` passes in both orders.
* **F3 (non-expansiveness at tiny h).** For $h$ below the rounding error of $T(|e|)$ the exact maps overshot by an ulp, and the JR `exp` family with $\rho_2>0$ returned `inf` for $|e|\gtrsim37/a$ ($\kappa$ rounds to 1, $\kappa^{-1}(1)=\infty$). Fix: clip to $|e_{k+1}|\le|e_k|$.
* **F4 (overflow → nan).** For $p>1$ terms, $|e|^{p-1}=\infty$ gave $\infty/\infty=$ nan in (C5) and in the Bernoulli-high exact map. Fix: algebraically identical forms $1/(1/y+c)$ and $v_+=e^{-k}/(1/v-c\,\mathrm{expm1}(-k))$ (the latter also removes the cancellation in $1-c\phi$).
* **F5 (silent fallback for zoo activations).** `peb`/`exact` recognised only `linear` and `sign_bi_power`, so `peb_euler` with the zoo `sign_power` silently ran the (non-finite-step) resolvent. Fix: verified decomposition of the 10 power-sum zoo members, plus the `predefined_exp` ↔ (JR-7) equivalence (§6).
* **F6 (names).** Scheme names now equal registry keys; `implicit_taylor_zhang(error_map="peb")` is now named `peb_taylor_zhang` instead of a misleading `implicit_taylor_zhang`; an unknown `error_map` raises in the factory.
* Stale core request: item 3 of the `consistent` section of `docs/spec/CORE_CHANGE_REQUESTS.md` (no duplicate check in `activations.register`; guarded registration of `power_sum`/`jr_predefined`) is obsolete. `activations.register` now raises on duplicates and this module no longer registers activations globally. That file is outside this area's edit permissions, so the item should be withdrawn by its maintainer.

### 11.4 Test meaningfulness

Mutation checks: removing the F3 clip makes 7 cases of `test_maps_sign_preserving_and_nonexpansive` fail; reverting the F4 form makes `test_maps_finite_for_huge_errors` fail. References used by the tests are independent of the implementation: DOP853 flows, quadrature, the closed forms printed in [PEB19] §1.1/§1.2 and [JR20] Eq. (2), [AL19] Thm 1 via `scipy.special.gamma`, and the settling-time formulas of `activations.py`.

## 12. Integration change (integrator, 2026-10-18)

`ConsistentZNN.desired_Edot` now passes `self.act.at(ctx.t)` to the error-map request, as `DesignFormulaZNN` does (see docs/spec/activations.md, `strict_predefined`). A time-varying activation is therefore frozen at the evaluation time instead of raising. For every time-invariant activation `at(t)` returns the activation itself, so the maps and all results above are unchanged. `strict_predefined` has no power terms or exact step, so `peb`/`exact` fall back to its resolvent, which is analytic before t_c, and warn once.
