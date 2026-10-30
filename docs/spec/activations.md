# Activation zoo (`znnbench/activations.py`)

Area: **activations**. Files: `znnbench/activations.py`, `tests/test_activations.py`, this document.

All activations enter the ZNN design formula

$$\dot E(t) = -\gamma\,\Phi(E(t)) \qquad(\text{element-wise }\Phi)$$

so each scalar error entry obeys $\dot e = -\gamma\,\Phi(e)$. Every class is odd and monotonically
non-decreasing. Each class has a `source` string (citation and DOI) and a `provenance` tag:

* **primary**: the formula was read in the original paper's full text.
* **secondary**: the original was paywalled. The formula was read in a later open-access paper that
  transcribes it. The transcribing paper is named below.
* **derived**: the item is a ZNN-Bench derivation, flagged **DERIVED** or **ASSUMPTION**.

## API (extends the original `Activation` base; existing API is unchanged)

| member | meaning |
|---|---|
| `__call__(e)` | $\Phi(e)$, element-wise, any shape |
| `slope0` | $\Phi'(0)$ (`np.inf` when $\Phi$ is not Lipschitz at 0, `0` for the power activation) |
| `jump0()` | $\Phi(0^+)$ (positive for activations that contain a sign term) |
| `prox(e, h)` | resolvent $y=(I+h\Phi)^{-1}(e)$. The inclusion $y+h\Phi(y)\ni e$ returns exactly 0 in the dead zone $\lvert e\rvert\le h\Phi(0^+)$. Analytic for `linear`, `power` (p=3, hyperbolic Cardano formula), `sign` (soft threshold) and `strict_predefined` before $t_c$. `power` with p≥5 uses monotone Newton. All other activations use the generic 80-step **geometric (log-scale) bisection** on $[\text{tiny},\lvert e\rvert]$, which has relative accuracy at every scale (changed by the verifier; see Verification). |
| `finite_time`, `convergence` | class of the continuous-time scalar dynamics: `exponential`, `asymptotic`, `finite`, `fixed`, `predefined`, `strictly-predefined`. It equals the source's claim except for `sign_bi_power` and `wsbp`, which are claimed finite-time but are **DERIVED** fixed-time (finite `uniform_bound`). |
| `claimed_convergence` | the class claimed by the source (equals `convergence` except for `sign_bi_power` and `wsbp`, where it is `finite`) |
| `upper_bound_time(e0, gamma)` | convergence-time bound **stated by the source**, else `None`. When `e0` is an array, the bound uses $\max_i\lvert e_{0,i}\rvert$, as every cited theorem does. |
| `uniform_bound(gamma)` | a bound valid for all $e_0$ (`inf` if none exists) |
| `settling_time(e0, gamma)` | **exact** settling time $T(e_0)=\frac1\gamma\int_0^{\lvert e_0\rvert}\frac{ds}{\Phi(s)}$ (quadrature after the substitution $s=\lvert e_0\rvert e^{-v}$; closed form where one exists). It is `inf` for asymptotic activations. This is the improper-integral characterisation that Xiao et al. (IEEE TII 2021) used to classify FTC and PTC. |
| `at(t)` | the activation frozen at time t (`self` for time-invariant activations) |
| `tight_uniform_bound(gamma)` | **DERIVED** $\sup_{e_0}T(e_0)$ of the bi-power part (Beta integral) for `sign_bi_power`, `wsbp`, `vaf`, `fixed_time` |
| `available()` | sorted list of keys |

$\operatorname{sig}^r(e) := \operatorname{sign}(e)\lvert e\rvert^r$ ("Lip$^r$" in Li et al.), with $\operatorname{sig}^r(0)=0$.

---

## 1. Exponential and asymptotic activations

### `linear` (exists, unchanged)
* Zhang, Jiang, Wang, IEEE TNN 13(5):1053–1063, 2002, doi:10.1109/TNN.2002.1031938. Also Zhang & Ge, IEEE TNN 16(6):1477–1490, 2005, doi:10.1109/TNN.2005.857946, activation (i).
* $\Phi(e)=e$, $e(t)=e_0e^{-\gamma t}$. Provenance: secondary, from the list of Zhang & Ge's four activations transcribed by Wang, Dai, Wang, ICITMI 2015, doi:10.2991/icitmi-15.2015.161.

### `power`
* Zhang–Jiang–Wang 2002 and Zhang & Ge 2005, activation (iii). Secondary (Wang et al. 2015; Zhou & Tan 2022, Table 1).
* $\Phi(e)=e^p$, with p an odd integer ≥ 3 (Zhou & Tan's table writes "k > 3"; Zhang & Ge use p ≥ 3, which is what is enforced). Default p = 3.
* $\Phi'(0)=0$, so the decay is algebraic: $e(t)=e_0\,(1+(p-1)\gamma e_0^{p-1}t)^{-1/(p-1)}$. The class is `asymptotic`.
* `prox`: for p = 3, $y^3+y/h-\lvert e\rvert/h=0$ gives $y=\frac{2}{\sqrt{3h}}\sinh\!\big(\tfrac13\operatorname{asinh}(1.5\lvert e\rvert\sqrt{3h})\big)$. For p ≥ 5 a Newton iteration starts at $\min(\lvert e\rvert,(\lvert e\rvert/h)^{1/p})$. It converges monotonically because $g$ is convex and increasing.

### `bipolar_sigmoid`
* Zhang & Ge 2005, activation (ii). Secondary (Wang et al. 2015). The unscaled form also appears in Zhou & Tan 2022, Table 1.
* $\Phi(e)=\dfrac{1-e^{-\xi e}}{1+e^{-\xi e}}=\tanh(\xi e/2)$. The source requires $\xi\ge 2$.
* Default ξ = 4. **ASSUMPTION:** by analogy with the power-sigmoid simulations (Wang et al. 2015 use p = 3, ξ = 4 for the power-sigmoid).
* `normalized=True` multiplies by $\frac{1+e^{-\xi}}{1-e^{-\xi}}$. This is the scaling used inside the power-sigmoid.
* $\Phi'(0)=\xi/2$, or $\frac{\xi}{2}\coth\frac{\xi}{2}$ when normalised.

### `power_sigmoid` (exists, verified)
* Zhang & Ge 2005, activation (iv). Secondary (Wang et al. 2015; Zhou & Tan 2022, Table 1).

$$\Phi(e)=\begin{cases}e^p,&\lvert e\rvert\ge1\\[2pt]\dfrac{1+e^{-\xi}}{1-e^{-\xi}}\cdot\dfrac{1-e^{-\xi e}}{1+e^{-\xi e}},&\lvert e\rvert<1\end{cases}\qquad \xi\ge2,\ p\ge3\ \text{odd}.$$

* The existing implementation was checked: $\tanh(\xi e/2)/\tanh(\xi/2)$ is identical, and $\Phi'(0)=\frac{\xi}{2}\coth\frac{\xi}{2}$. The docstring was corrected from "ξ > 2" to the source's "ξ ≥ 2". Odd p is now enforced.
* Defaults p = 3, ξ = 4: the values of the power-sigmoid simulations in Wang et al. 2015 (verified); Zhang & Ge 2005 itself was not read.

### `power_sum`
* Yang & Zhang, Neural Comput. Appl. 22:175–185, 2013, doi:10.1007/s00521-011-0692-5.
* $\Phi(e)=\sum_{k=1}^{N}e^{2k-1}$, with integer $N\ge2$.
* **Provenance: secondary and weak.** The full text was not accessible. The form "sum of odd powers, integer N > 1" comes from a search-engine summary of the Jin–Li–Liao–Zhang survey (Neurocomputing 267, 2017). **ASSUMPTION:** default N = 3.

### `hyperbolic_sine`
* Zhang, Jin, Ke, Comput. Sci. Inf. Syst. 9(4):1603–1625, 2012, doi:10.2298/CSIS120121043Z, Sec. 2. **Primary** (open access).
* $\Phi(e)=\tfrac12e^{\xi e}-\tfrac12e^{-\xi e}=\sinh(\xi e)$, with $\xi\ge1$.
* Default ξ = 3. The paper's simulations use ξ = 3, 5, 7.
* $\Phi'(0)=\xi$. Exact scalar solution: $\tanh(\xi e/2)=\tanh(\xi e_0/2)e^{-\gamma\xi t}$ (DERIVED; used in the tests).

## 2. Finite-time activations

### `sign`
* Zhang, Ding, Qiu, Zhang, Li, Inf. Process. Lett. 124:30–34, 2017, doi:10.1016/j.ipl.2017.04.008 ("signum-function array activated ZNN").
* $\Phi(e)=\operatorname{sign}(e)$. Exact settling time $T=\lvert e_0\rvert/\gamma$ (DERIVED; the paper's own statement was not retrieved).
* `prox` is the soft threshold $\operatorname{sign}(e)\max(\lvert e\rvert-h,0)$. This is the finite-step and chattering baseline.

### `sign_power`
* Zhang, Chen, Li et al., CAAI Trans. Intell. Technol. 6(4):394–404, 2021, doi:10.1049/cit2.12019. Eqs. (8) and (11) (SSZNN); Theorem 3. **Primary.**
* $\Phi(e)=\operatorname{sig}^r(e)$, $0<r<1$. Bound (exact): $T=\lvert e_{\max}(0)\rvert^{1-r}/(\gamma(1-r))$.
* **ASSUMPTION:** default r = 0.5.

### `sign_bi_power` (exists, verified)
* Li, Chen, Liu, Neural Process. Lett. 37:189–205, 2013, doi:10.1007/s11063-012-9241-1.
* **Secondary.** Transcribed identically by Hu et al., Neurocomputing 418:79–90, 2020, doi:10.1016/j.neucom.2020.08.037, eq. (10); by Zhang et al. 2021, eqs. (7)–(8); and by Wang et al. 2015, eqs. (3)–(4).

$$\Phi(e)=\tfrac12\operatorname{sig}^r(e)+\tfrac12\operatorname{sig}^{1/r}(e),\qquad 0<r<1.$$

* Source bound, as quoted by Hu et al. 2020, Thm. 2, and Wang–Hua–Khan 2025: $t_f\le \dfrac{2\lvert e_{\max}(0)\rvert^{1-r}}{\gamma(1-r)}$.
* **DERIVED** uniform (fixed-time) bound from both terms: $t_f\le\dfrac{2(1+r)}{\gamma(1-r)}$, and the tight one $\sup_{e_0}T=\frac{2\pi}{\gamma(1/r-r)\sin(\pi m)}$, $m=\frac{1-r}{1/r-r}=\frac{r}{1+r}$ (`tight_uniform_bound`; 4.837/γ for r = ½). The source bound becomes very loose for large $e_0$. For example, with r = 0.5, $e_0=10^3$ and γ = 1, the source bound is 126.5 while the exact value is 4.83 (**corrected by the verifier**: the implementer's report said 2.4, which drops the factor ½). The source claims finite-time convergence; `convergence = "fixed"` is a ZNN-Bench derivation and `claimed_convergence = "finite"`.
* **"Li activation function".** Liao & Zhang (Neurocomputing 133:512–522, 2014, doi:10.1016/j.neucom.2013.12.001), Guo & Zhang (Appl. Soft Comput. 24:158–168, 2014, doi:10.1016/j.asoc.2014.06.045) and Wang et al. 2015 use this name for exactly this function, with the factor ½. It is therefore **not a distinct activation** and has no separate key. Some later papers drop the ½ (Liao et al., Front. Neurorobot. 2022, implemented as `LiActivation` in `models/variants.py`). That version equals `sign_bi_power` with gain 2γ.

### `sfaf`: simplified finite-time activation (SFAF / SSBP-AF / "combined AF")
* Xiao, Zhang, Li, IEEE TSMC-Syst. 49(11):2210–2220, 2019, doi:10.1109/TSMC.2018.2836968.
* Secondary, via Hu et al. 2020, eq. (11) and Thm. 2. **Primary** confirmations of the special cases:
  * Zuo, Li, Xiao, Li, IEEE TNNLS 33(12):7695–7705, 2022, doi:10.1109/TNNLS.2021.3086500, eq. (6) and Thm. 1, with b1 = b2 = 1, citing Xiao et al., IEEE TII 14(1), 2018, doi:10.1109/TII.2017.2717020.
  * Zhang et al. 2021, Thm. 4 (CZNN, $\kappa_1\mathrm{Lf}^\tau+\kappa_2e$).

$$\Phi(e)=b_1\operatorname{sig}^r(e)+b_2e,\qquad T=\frac{1}{\gamma b_2(1-r)}\ln\!\Big(1+\frac{b_2\lvert e_0\rvert^{1-r}}{b_1}\Big)\ \text{(exact)}.$$

* Defaults b1 = b2 = 1 (Zuo et al.). **ASSUMPTION:** r = 0.5.

### `wsbp`: weighted sign-bi-power (tunable SBP)
* Xiao, Neurocomputing 173:1983–1988, 2016, doi:10.1016/j.neucom.2015.08.031. Secondary: eq. (6) of Chai et al., IJCIS 13(1):663–671, 2020, doi:10.2991/ijcis.d.200527.001.
* Chai et al. eq. (6) (read on the rendered PDF by the verifier): $\phi(x)=\lambda_1\operatorname{sgn}^k(x)+\lambda_2\operatorname{sgn}^{1/k}(x)+\lambda_3x$ with $\lambda_{1,2,3}>0$, $k\in(0,1)$, citing Xiao 2016. **ASSUMPTION:** ZNN-Bench relaxes $\lambda_3>0$ to $k_3\ge0$ so that SBP is the special case $k_1=k_2=\tfrac12$, $k_3=0$.
* Also published as "Wsbp" by Lv, Xiao, Tan, Yang, Neurocomputing 314:310–315, 2018, doi:10.1016/j.neucom.2018.06.057. That paper was not read.
* The sources claim finite-time convergence (`claimed_convergence = "finite"`); the DERIVED uniform bound makes it fixed-time (`convergence = "fixed"`). `tight_uniform_bound` is the Beta integral of the $k_1,k_2$ terms (exact sup when $k_3=0$).

$$\Phi(e)=k_1\operatorname{sig}^r(e)+k_2\operatorname{sig}^{1/r}(e)+k_3e,\quad 0<r<1,\ k_{1,2}>0,\ k_3\ge0.$$

* `upper_bound_time = None`: the source's bound was not retrieved. **DERIVED** uniform bound: $\frac{1}{\gamma k_1(1-r)}+\frac{r}{\gamma k_2(1-r)}$.
* **ASSUMPTION:** defaults k = (1, 1, 1), r = 0.5.
* Shen et al. (NPL 42:763–784, 2015, doi:10.1007/s11063-014-9397-y) proposed a "tunable" SBP with a linear term. Its exact form was **not verified**, so it was not implemented separately.

### `saturated_sign_power`
* Zhang, Chen, Li et al. 2021 (CAAI TIT), Sec. 3.4 (SZNN), Theorem 5. **Primary.**
* The paper defines $C(e)=\operatorname{sat}_{\eta}(\gamma\,\mathrm{Lf}^r(e))$ and model (17) with $-\gamma C$. Its proof, however, uses $\dot e=-\eta$ in saturation. ZNN-Bench follows the proof's reading: $\dot e=-C(e)=-\gamma\Phi(e)$ with

$$\Phi(e)=\operatorname{sign}(e)\min(\lvert e\rvert^r,s),\qquad s=\eta/\gamma.$$

* **DERIVED** exact bound, with $e_s=s^{1/r}$:
  * $T=\frac{\lvert e_0\rvert^{1-r}}{\gamma(1-r)}$ if $\lvert e_0\rvert\le e_s$;
  * otherwise $T=\frac{\lvert e_0\rvert-e_s}{\gamma s}+\frac{e_s^{1-r}}{\gamma(1-r)}$.
* **Source inconsistency flagged:** the printed bound $\lvert e_{\max}(0)\rvert/\eta_1-1+\eta_1^{1-r}/(\gamma(1-r))$ assumes the saturation switches at $\lvert e\rvert=\eta_1$. That contradicts the definition.
* **ASSUMPTION:** defaults r = 0.5, s = 1.

## 3. Fixed-time and "predefined-time" activations

### `vaf`: versatile activation function
* Xiao, Zhang, Dai, Chen, Yang, Li, Liao, Ding, Li, Neural Netw. 117:124–134, 2019, doi:10.1016/j.neunet.2019.05.005 (NNTZNN).
* Secondary: Zhang & Yin, AIMS Math. 7(2):2266–2280, 2022, doi:10.3934/math.2022129, eq. (4.1), Lemma 3, Thm. 2 and eq. (4.6), where the function is called "universal" and was "first presented in [Xiao 2019]". Also Zhou & Tan 2022, Table 1, and Wang, Hua, Khan, Biomimetics 10:279, 2025, doi:10.3390/biomimetics10050279.

$$\Phi(e)=(a_1\lvert e\rvert^{\eta}+a_2\lvert e\rvert^{\omega})\operatorname{sign}(e)+a_3e+a_4\operatorname{sign}(e),\quad 0<\eta<1<\omega,\ a_{1,2}>0,\ a_{3,4}\ge0,$$

$$t_c\le\frac{1}{\gamma a_1(1-\eta)}+\frac{1}{\gamma a_2(\omega-1)}.$$

* The same form is activation F1 of Li, Liao, Xiao, Lu 2019 (below), with the same bound.
* **ASSUMPTION:** defaults $a_i=1$, η = 0.5, ω = 2, taken from Zhang & Yin 2022. Xiao et al.'s own values were not retrieved.
* Additional method `tight_uniform_bound` (**DERIVED**): $\sup_{e_0}T=\frac{(a_1/a_2)^m\pi}{\gamma a_1(\omega-\eta)\sin(\pi m)}$ with $m=\frac{1-\eta}{\omega-\eta}$. This is the Beta integral, i.e. the tight fixed-time estimate of Parsegov et al. 2012 and Aldana-López et al. 2019 with k = 1. The $a_3$ and $a_4$ terms only shorten T.

### `fixed_time`: bi-power with linear term
* Jin, Zhu, Zhao, Chen, Appl. Soft Comput. 130:109691, 2022, doi:10.1016/j.asoc.2022.109691. **Secondary and weak:** the formula comes from the Wang–Hua–Khan 2025 Table 1 entry "$(b_1\lvert e\rvert^\kappa+b_2\lvert e\rvert^\eta)\operatorname{sgn}(e)+b_3e$, fixed-time".
* $\Phi(e)=(k_1\lvert e\rvert^{p}+k_2\lvert e\rvert^{q})\operatorname{sign}(e)+k_3e$, with $0<p<1<q$. This is the VAF with $a_4=0$.
* **ASSUMPTION:** the bound $\frac{1}{\gamma k_1(1-p)}+\frac{1}{\gamma k_2(q-1)}$ is the VAF bound (Polyakov-type); the ASOC paper's own statement was not read. The class also provides `tight_uniform_bound`.

### `pvaf1`, `pvaf2`: power-versatile activations
* Zhou & Tan, Bull. Pol. Acad. Sci. Tech. Sci. 70(3):e141307, 2022, doi:10.24425/bpasts.2022.141307. **Primary** (open PDF).

$$\text{PVAF-1 (6):}\ \Phi(e)=(a\lvert e\rvert^p+b\lvert e\rvert^q)^k\operatorname{sign}(e)+c_1e+c_2\operatorname{sign}(e),\quad a,b,p,q,k>0,\ pk>1,\ qk<1,$$

$$t_s\le\frac{1}{\gamma a^k(pk-1)}+\frac{1}{\gamma b^k(1-qk)}\quad\text{(Lemma 1, eq. (13); Thm. 1)}.$$

* k = 1 recovers the VAF.

$$\text{PVAF-2 (7):}\ \Phi(e)=(a\lvert e\rvert^p+b)^k\operatorname{sign}(e)+d_1e+d_2\operatorname{sign}(e),\quad pk>1,$$

$$t_s\le\frac{1}{\gamma b^k}\Big(\frac ba\Big)^{1/p}\Big(1+\frac{1}{pk-1}\Big)\quad\text{(Lemma 2, eq. (15))}.$$

* The PVAF-2 bound was garbled in the implementer's text extraction and re-derived by splitting at $\lvert e\rvert=(b/a)^{1/p}$. **Verifier:** eqs. (6), (7), (13), (15), Theorem 1 and Table 1 were read on the rendered PDF pages; the code matches symbol by symbol. The gain enters as $a,b\to\gamma^{1/k}a,\gamma^{1/k}b$ (proof of Thm. 1), i.e. $a^k,b^k\to\gamma a^k,\gamma b^k$. The paper's simulations vary $k\in\{0.5,1,2\}$ and print no $a,b,p,q$.
* **ASSUMPTION:** defaults (a, b, p, q, k, c1, c2) = (1, 1, 1, 0.25, 2, 1, 1) and (a, b, p, k, d1, d2) = (1, 1, 1, 2, 1, 1).

### `predefined_exp`: exponential predefined-time activation (F2)
* Li, Liao, Xiao, Lu, Neurocomputing 337:262–273, 2019, doi:10.1016/j.neucom.2019.01.072. This is the first predefined-time ZNN.
* Secondary: Wang–Hua–Khan 2025, Sec. 3.1. In this code $k_{1,2,3}$ stand for the paper's $\gamma_{1,2,3}$.

$$\Phi(e)=\frac{k_1}{p}\exp(\lvert e\rvert^p)\lvert e\rvert^{1-p}\operatorname{sign}(e)+k_2e+k_3\operatorname{sign}(e),\ 0<p<1,\qquad t_c\le\frac{1}{\gamma k_1}.$$

* **DERIVED:** with $k_2=k_3=0$ the settling time is exactly $(1-e^{-\lvert e_0\rvert^p})/(\gamma k_1)$. This is the Sánchez-Torres et al. predefined-time system.
* **ASSUMPTION:** defaults p = 0.5, k = (1, 1, 1).
* Variants with p multiplying instead of dividing appear in later papers (e.g. Xiao et al., Neurocomputing 526, 2023; Liu, Zhao, Jin 2024). They were not implemented.

### Xiao, Cao, Dai, Jia, Tan, IEEE TII 17(7):4724–4732, 2021, doi:10.1109/TII.2020.3021438 (NOT reproduced)
* Only the abstract was accessible (IEEE, closed access; no preprint found).
* The paper derives FTC and PTC criteria for $\dot e=-\gamma\Phi(e)$ from the limit comparison test for improper integrals, and proposes new AFs.
* The criterion corresponds to `settling_time`, which evaluates $\frac1\gamma\int_0^{\lvert e_0\rvert}ds/\Phi(s)$:
  * FTC ⇔ the integral converges at 0;
  * PTC/fixed-time ⇔ $\int^{\infty}ds/\Phi(s)<\infty$ in addition.
* **The new AFs of that paper are not implemented, because their formulas could not be verified.**

## 4. Time-varying activation

### `strict_predefined` (SPTC-NT-ZNN activation)
* Li, Guo, Ma, Pan, IEEE TIE 71(1):798–809, 2024, doi:10.1109/TIE.2023.3241393.
* Secondary: Yang et al., arXiv:2503.01857, Table 1, and Wang–Hua–Khan 2025.

$$\Phi(e,t)=\begin{cases}\dfrac{e}{t_c-t},&t<t_c\\[4pt] e+\lvert e\rvert^p\operatorname{sign}(e)+\xi\operatorname{sign}(e),&t\ge t_c\end{cases}\qquad 0<p<1,\ \xi\ge0.$$

* With $\dot e=-\gamma\Phi$: $e(t)=e_0\big(\tfrac{t_c-t}{t_c}\big)^\gamma$, so $e(t_c)=0$ for every $e_0$ and every γ > 0 ("strict": the settling time is exactly $t_c$).
* The model-level implementation is `sptc_nt_znn` in `models/variants.py` (another area). This class exists for activation-level studies through `at(t)` or `act(e, t=...)`; a direct call without t raises. *Integration change:* `DesignFormulaZNN` (and its subclasses, `GNN`, `hntznn`, `znn_consistent`) now freezes a time-varying activation at the evaluation time with `act.at(ctx.t)`, so `make_model("znn", activation="strict_predefined")` runs the SPTC-NT-ZNN design formula with constant gain instead of raising at the first step. For time-invariant activations `at(t)` returns the activation itself, so nothing else changes.
* Guard: $t\ge t_c-10^{-12}\max(1,t_c)$ counts as post-$t_c$, as in `models/gains.py`. `slope0` = $1/t_c$, the value at t = 0.
* **ASSUMPTION:** defaults $t_c=1$, p = 0.5, ξ = 0 (ξ depends on the noise level).
* **Discrete-time observation (tested):** the last fixed step samples $t=t_c$ on the switched branch, which leaves an $O(\tau)$ remainder. For γ = 1, RK4 gives exactly $e_0\tau/(6t_c)$. The post-$t_c$ finite-time branch then removes it down to the RK4 chattering floor $(\gamma\tau)^{1/(1-p)}$.

## Not implemented (and why)
* **Segmented FTNTZNN activation** (Xiao et al., IEEE TNNLS 35:10503, 2024, doi:10.1109/TNNLS.2023.3242313). The only transcription available (Wang–Hua–Khan 2025) is internally inconsistent: continuity at |z| = 1 and the |z| vs |z_l| notation do not agree.
* **"Finitely-activated" saturated AF** (Sun et al., NCA 37, 2025). The transcription is garbled.
* Fuzzy-adaptive and other recent AFs (sinh/sum mixtures, $h\sin(\lvert x\rvert^n)$, ...). These are out of scope or unverified.
* A separate `li` key. See `sign_bi_power` above.

## Tests (`tests/test_activations.py`, 246 tests, about 5 s, well under 100 MB)
1. **Registry/API:** the registry keys defined in `activations.py` are exactly the 18 expected keys (keys that sibling modules may add to the shared registry are ignored, so the file is order-independent); `describe`, `source`, `provenance`, `convergence` and `claimed_convergence` are consistent; duplicate keys are refused; invalid parameters (even p, r ∉ (0,1), pk ≤ 1, …) raise `ValueError`.
2. **Formulas:** spot values at 10 points are re-transcribed **independently** from the equations above, with non-default parameters. Equivalence checks:
   * SBP = WSBP(½, ½, 0);
   * SSBP-AF $q+\operatorname{sig}^r(q)$ (Zuo eq. 6);
   * VAF(a4 = 0) = `fixed_time`;
   * PVAF-1(k = 1) = VAF;
   * normalised bipolar sigmoid = the inner branch of power-sigmoid;
   * the branches of `strict_predefined`.
3. **Structure:** oddness, monotonicity, $\Phi(0)=0$; `slope0` against a central difference, with a blow-up check for non-Lipschitz activations.
4. **prox:** $y+h\Phi(y)=e$ to 1e-10 relative outside the dead zone, $y=0$ exactly inside it. Tested with a scalar h, with batched h of shape (B,1), and for `strict_predefined` before and after $t_c$. A second test uses strongly non-Lipschitz parameters (exponents 0.02–0.1, $h$ up to $10^4$, $\lvert e\rvert$ down to $10^{-9}$), where the resolvent lies 30–300 orders of magnitude below $\lvert e\rvert$, and requires a **relative** residual ≤ 1e-12 (the pre-verification linear bisection gave relative residuals up to $3\cdot10^{11}$ there). The analytic proxes (power p = 3/5/9, sign, linear) agree with 200-step bisection.
5. **Asymptotic dynamics:** RK4 (τ = 2.5e-4) on $\dot e=-\gamma\Phi(e)$ reproduces the closed-form solutions of linear, power (p = 3), sinh, bipolar sigmoid and power-sum (N = 2) to 1e-7. Near 0 the rate is $\gamma\Phi'(0)$. **Discrete-time:** explicit Euler $e_{k+1}=e_k-h\Phi(e_k)$ is locally contractive at 0 for $h=0.9\cdot2/\Phi'(0)$ and not for $h=1.1\cdot2/\Phi'(0)$.
6. **Finite/fixed/predefined bounds:** for $e_0\in[10^{-4},10^4]$, `settling_time` ≤ `upper_bound_time` (when the source gives one) and ≤ `uniform_bound`, with $T\propto1/\gamma$; the same at **random valid non-default parameters** (4 draws per key, 4 values of $e_0$, random γ), including `tight_uniform_bound`. Fixed and predefined activations satisfy $T(10^{12})\le$ `uniform_bound`. Closed forms are checked against quadrature: sign, sign-power, SFAF log formula, saturated, predefined-exp exact formula, PVAF-1/PVAF-2 bounds, and the tight bi-power bound equal to $\sup T$ (fixed_time, SBP: $2\pi/(1.5\sin(\pi/3))$ for r = ½, WSBP with $k_3=0$).
7. **RK4 reaches ~0 by the bound:** τ = 2e-4, with γ scaled so that the bound for $\max\lvert e_0\rvert$ is 1 s. $\lvert e\rvert\le2\cdot10^{-3}$ is first reached no later than the stated bound (or the exact $T$ when the source gives none). It is reached no earlier than $T(e_0)-T(\text{tol})$, and the error stays below tol afterwards.
8. **Strict predefined-time:** the trajectory follows $e_0((t_c-t)/t_c)^\gamma$ for γ ∈ {1, 3} and $e_0\in\{-5, 0.1, 2, 50\}$; the $O(\tau)$ remainder at $t_c$ is checked as described in Section 4.
9. **Integration:** every time-invariant key runs inside `make_model("znn", activation=key)` with `euler` and `implicit_euler_err` (prox) on `linear_system` (n = 3, 20 steps).

## Discrete-time note (explicit Euler step-size limit)
For the Lipschitz activations, explicit Euler on the error dynamics, $e_{k+1}=e_k-h\Phi(e_k)$ with $h=\gamma\tau$, is locally stable at 0 iff $h\,\Phi'(0)<2$ (tested). With the defaults: `linear` h < 2, `bipolar_sigmoid` (ξ = 4) h < 1, `power_sigmoid` (p = 3, ξ = 4) h < 0.964, `hyperbolic_sine` (ξ = 3) h < 2/3, `power_sum` h < 2. So part of the default benchmark range $h\in(0,2)$ is locally unstable for the nonlinear Lipschitz activations, independently of their continuous-time class. Far from 0 the power branches add the global restriction $h\lvert e\rvert^{p-1}\lesssim2$.

## Verification (adversarial review)
Performed by the verifier after the implementer's report. Every registered key was compared symbol by symbol (formula, signs, exponents, parameter conditions, defaults, bound) against the source that could be opened in this session. DOIs, titles, volumes and pages of the 25 main cited works (all sources of registered keys) were checked against Crossref.

| key | checked against | result |
|---|---|---|
| `linear`, `power`, `bipolar_sigmoid`, `power_sigmoid` | Wang, Dai, Wang 2015 (ICITMI, open PDF, read): transcription of Zhang & Ge 2005 activations (i)–(iv) and its simulations (p = 3, ξ = 4); Zhou & Tan 2022 Table 1 (rendered PDF) | match (secondary: Zhang & Ge 2005 and Zhang–Jiang–Wang 2002 are closed access) |
| `power_sum` | no open source found; only the Yang & Zhang 2013 title/DOI (Crossref; the Biomimetics 2025 review cites it without a formula) | unverifiable; formula kept as **secondary/weak** |
| `hyperbolic_sine` | Zhang, Jin, Ke 2012 (ComSIS, open PDF, read): $f(e)=\exp(\xi e)/2-\exp(-\xi e)/2$, $\xi\ge1$, simulations ξ = 3, 5, 7, γ = 1 | match (primary) |
| `sign` | IPL 2017 title/DOI only (closed) | formula trivially consistent with the title; settling time is an exact derivation |
| `sign_power`, `saturated_sign_power` | Zhang et al. 2021 (CAAI TIT) could **not** be opened in this session (Wiley/IET return HTTP 403) | unverifiable here; the exact settling times were re-derived and checked by quadrature |
| `sign_bi_power` | Wang–Hua–Khan 2025 (PMC full text, read): $\frac12(\mathrm{Lip}^a+\mathrm{Lip}^{1/a})$ and bound $\max\{2\lvert e^\pm(0)\rvert^{1-a}/(\varrho(1-a))\}$; Wang et al. 2015 "Li activation function" eqs. (3)–(4) | match (secondary); exact-value example in the spec corrected (4.83, not 2.4) |
| `sfaf` | Hu et al. 2020 and Zuo et al. 2022 could not be opened (Newcastle repository record has no file; TNNLS closed) | unverifiable here; log settling-time formula re-derived exactly and checked by quadrature |
| `wsbp` | Chai et al. 2020 eq. (6) and conditions (rendered PDF, read) | match, except that ZNN-Bench allows $k_3=0$ (flagged ASSUMPTION) |
| `vaf` | Zhang & Yin 2022 (AIMS, open PDF, read): eq. (4.1), conditions $0<p<1$, $q>1$, $k_{1,2}>0$, $k_{3,4}\ge0$, Lemma 3, eq. (4.6) bound $\frac1{\gamma k_1(1-p)}+\frac1{\gamma k_2(q-1)}$, simulation values $k_i=1$, p = 0.5, q = 2; Zhou & Tan Table 1; Wang–Hua–Khan 2025 | match (secondary; the Xiao et al. 2019 repository PDF returned HTTP 403) |
| `fixed_time` | Wang–Hua–Khan 2025 Table 1 (PMC, read): $(b_1\lvert e\rvert^\kappa+b_2\lvert e\rvert^\eta)\operatorname{sgn}(e)+b_3e$, $b_i,\kappa,\eta>0$, fixed-time, refs. Jin et al. 2022 ASOC and Sun et al. 2023 | match (secondary/weak). The code's $0<p<1<q$ is the necessary condition for fixed time; the review's $b_3>0$ is relaxed to $k_3\ge0$ |
| `pvaf1`, `pvaf2` | Zhou & Tan 2022 (rendered PDF pages 2–3, read): eqs. (6), (7), (12)–(15), Thm. 1 | match (primary), including the PVAF-2 bound (15) |
| `predefined_exp` | Wang–Hua–Khan 2025 (PMC, read): $F_2(x)=\gamma_1\exp(\lvert x\rvert^p)\lvert x\rvert^{1-p}\operatorname{sign}(x)/p+\gamma_2x+\gamma_3\operatorname{sign}(x)$, $t_c\le1/(\lambda\gamma_1)$; F1 = VAF form with bound $\frac1{\lambda\kappa_1(1-p)}+\frac1{\lambda\kappa_2(q-1)}$ | match (secondary) |
| `strict_predefined` | Yang et al., arXiv:2503.01857v1, Table 1 (rendered PDF, read), gain $\gamma(t)=\gamma$; Wang–Hua–Khan 2025 | match (secondary) |

**Independent numerical checks** (scratch scripts, not part of the suite):
* `settling_time` against a separate direct quadrature of $\int_0^{a}ds/\Phi(s)$ for all 11 finite-time keys × 25 random valid parameter sets × 5 initial errors: maximal relative difference 2.3e-10. No violation of any `upper_bound_time`, `uniform_bound` or `tight_uniform_bound`.
* Beta-integral `tight_uniform_bound` against 30-digit mpmath quadrature (smooth substitution $s=e^x$) for 60 random bi-power parameter sets: maximal relative error 2.9e-16.
* Closed-form solutions used by the tests (sinh, bipolar sigmoid, power, power-sum, strict predefined, RK4 remainder $e_0\tau/(6t_c)$) re-derived by hand.
* `slope0` against a central difference at non-default parameters for the four nonlinear Lipschitz activations.

**Bugs found and fixed**
1. **Generic `prox` lost all accuracy for strongly non-Lipschitz activations.** The linear bisection on $[0,\lvert e\rvert]$ stalls at $\lvert e\rvert2^{-80}$, but for small exponents or large $h$ the root lies 30–300 orders of magnitude lower. Example: `fixed_time(p=0.07, q=3.97, …)`, $h=10^4$ gave $y\approx10^{-23}$ instead of $10^{-34}$, a residual of $\sim5\cdot10^{2}$; relative residuals reached $3\cdot10^{11}$ with `sfaf(r=0.06)` on $\lvert e\rvert\in[10^{-9},233]$. Replaced by geometric bisection with the same API and cost; regression test added. The effect on ZNN trajectories was small (both values ≈ 0 relative to $e$), but the resolvent contract was violated.
2. **Order-dependent test file.** `tests/test_activations.py` iterated the whole shared registry, so a sibling module that registered keys at import time (as `discretizations/consistent.py` did before its own fix) made `test_registry_contains_zoo` fail when run after `tests/test_consistent.py` (reproduced). The tests now only iterate the classes defined in `activations.py`.
3. **Mislabelled convergence class.** The API documentation said `convergence` is the class the source claims, but `sign_bi_power` and `wsbp` were labelled `fixed` although their sources claim finite-time. Added `claimed_convergence` and documented `convergence` as the (possibly DERIVED-stronger) class; tested.
4. **Wrong number in the spec:** the SBP example "source bound 126 vs exact 2.4" is corrected to 126.5 vs 4.83 (tested).
5. Added `tight_uniform_bound` to `sign_bi_power` and `wsbp` (the report said it existed "for the bi-power families"), and tests at random parameters.

**Still unverified (sources not accessible in this session):** Zhang et al. 2021 CAAI TIT (`sign_power`, `saturated_sign_power`, the SFAF special case), Hu et al. 2020 and Zuo et al. 2022 (`sfaf`), Yang & Zhang 2013 (`power_sum`), Jin et al. 2022 ASOC (`fixed_time`), and the closed originals listed in the implementer's report. The web-search budget of the session was exhausted, so no further open copies could be located.
