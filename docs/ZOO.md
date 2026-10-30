# ZNN-Bench zoo: catalogue of registered components

This is the list of every key a ZNN-Bench config can name, with a one-line description, the source and its DOI. The equations, defaults, every ASSUMPTION and the verification log for each key are in the area specs under `docs/spec/`, linked in the last column. `tests/test_integration.py::test_zoo_catalogue_covers_every_registered_key` checks that every key registered at import time is listed here.

```python
import znnbench
znnbench.available()                     # problems, models, discretizations, noises
from znnbench import activations; activations.available()
```

**Counts (2026-10-18, after the gap-fill round):** 13 problems, 18 models, 18 registered activations (plus 3 model-local ones), 42 discretizations, 11 noises, 5 wrapper classes, 6 gain schedules. A structured per-step FLOP model for every registered problem family, model and discretization is in `znnbench/costs.py` ([costs](spec/costs.md)).

**Check column**, i.e. what the equations were checked against during verification:

| code | meaning |
|---|---|
| P | the primary full text |
| S | a secondary open restatement, or, for difference formulas, the transcription plus an exact check of the order (moment) conditions |
| U | not checked against any text: the source was not accessible and the formula comes from a weak secondary source or an abstract |
| C | ZNN-Bench construction or benchmark extension, built on the cited works |
| T | classical textbook method |

The h = gamma*tau convention and the meaning of `expected_order` are in `docs/DESIGN.md` and `docs/spec/discretizations.md`.

## Problems (`make_problem(key, **kw)`)

Default instances are small (n = 4 where n applies). x is the row-major flattening of the unknown matrix.

| key | description | source | DOI | check | spec |
|---|---|---|---|---|---|
| `linear_system` | A(t)x = b(t), E = A x - b, smooth A(t) with controlled condition number (`_tv.TVMatrix`) | ZNN-Bench reference problem | — | C | DESIGN.md |
| `matrix_inversion` | E = A(t)X - I; capability `matrix_inverse`; `example='zhang_ge_2005'` | Zhang & Ge, IEEE TNN 16(6) 2005, eqs. (1)-(4), Sec. V | 10.1109/TNN.2005.857946 | P | [problems_matrix](spec/problems_matrix.md) |
| `matrix_inversion_right` | E = X A(t) - I (dual form); capability `matrix_inverse_right` | Živković & Stanimirović, Facta Univ. Ser. Math. Inform. 32(5) 2017, eq. (1.2) (stated for constant A; time-varying use is an ASSUMPTION) | printed DOI 10.22190/FUMI1705799Z is not registered; see spec | P | [problems_matrix](spec/problems_matrix.md) |
| `pseudoinverse` | full-rank Moore-Penrose inverse, left (AᵀA X - Aᵀ) or right (X A Aᵀ - Aᵀ) form | Zhang, Yang, Tan, Cai, Computing 92(2) 2011; form transcribed from Chai et al., IJCIS 13(1) 2020 | 10.1007/s00607-010-0133-9; 10.2991/ijcis.d.200527.001 | S | [problems_matrix](spec/problems_matrix.md) |
| `sylvester` | A(t)X - X B(t) + C(t) = 0 | Zhang, Jiang, Wang, IEEE TNN 13(5) 2002; form from Zhao et al., Front. Phys. 11 2023 | 10.1109/TNN.2002.1031938; 10.3389/fphy.2023.1133745 | S | [problems_matrix](spec/problems_matrix.md) |
| `lyapunov` | Aᵀ(t)X + X A(t) + Q(t) = 0; `example='hu_2020'` | Hu, Li, Li, Li, Xiao, Neurocomputing 418 2020, eqs. (1)-(4), Sec. 5 | 10.1016/j.neucom.2020.08.037 | P | [problems_matrix](spec/problems_matrix.md) |
| `complex_inversion` | complex A(t)X = I in the real embedding [[Ar, -Ai], [Ai, Ar]]; capability `complex_embedding` | Zhang, Li, Li, AMC 217(24) 2011 (complex E); Ding et al., Front. Neurorobot. 11 2017, eqs. (3)-(5) (embedding) | 10.1016/j.amc.2011.04.085; 10.3389/fnbot.2017.00045 | P | [problems_matrix](spec/problems_matrix.md) |
| `nonlinear_equations` | f(x, t) = 0: `guo2018` 4-D system (default; domain x1 > 0) or `manufactured` (any n) | Guo, Xu, Li, Nie, Shao, IEEE TII 14 2018; transcribed from Ma & Huang, Front. Neurorobot. 19 2025, Eq. (33) (Table 1 reproduced) | 10.1109/TII.2017.2787729; 10.3389/fnbot.2025.1576473 | S | [problems_opt](spec/problems_opt.md) |
| `tv_unconstrained_opt` | min_x f(x, t), E = ∇f, J = ∇²f; `zhang2021` (default), `quadratic`, `manufactured`; capability `gradient_field` | objective of benchmark (18) of Zhang, Chen, Li, Stanimirović, Zhang, Jin, CAAI TIT 6 2021 (constraints dropped); model class of Jin & Zhang, IEEE TNNLS 26 2015 | 10.1049/cit2.12019; 10.1109/TNNLS.2014.2342260 | P | [problems_opt](spec/problems_opt.md) |
| `tvqp_eq` | equality-constrained TV-QP in KKT form, state [x; λ] | Zhang & Li, Phys. Lett. A 373 2009 (formulation); data of CAAI TIT 2021 benchmark (18) | 10.1016/j.physleta.2009.03.011; 10.1049/cit2.12019 | P | [problems_opt](spec/problems_opt.md) |
| `tvqp_ineq` | adds C(t)x ≤ d(t) through the perturbed Fischer-Burmeister function, state [x; λ; μ]; `zhang2021`, `yang2025`, `random` | Zhang et al., CAAI TIT 6 2021, Def. 1, Lemma 2; Yang, Wang, Voyles, Ma, Tsinghua Sci. Technol. 31(3) 2026, Eqs. (2)-(7) | 10.1049/cit2.12019; 10.26599/TST.2024.9010202 | P | [problems_opt](spec/problems_opt.md) |
| `robot_puma560` | position tracking with a 6-DOF PUMA 560 (standard DH), paths `circle`/`lissajous`/`plum`/`heart`; `dh='corke'` or `'jin2017'` | Corke, Robotics, Vision and Control, 2017 (`mdl_puma560`); Jin, Li, La, Luo, IEEE TIE 2017, Table I and Sec. VI-B; paths from Yang et al., Neural Netw. 186 2025, eqs. (29)-(31) | 10.1007/978-3-319-54413-7; 10.1109/TIE.2017.2674624; 10.1016/j.neunet.2025.107279 | P | [problems_robot](spec/problems_robot.md) |
| `robot_panda` | position tracking with the 7-DOF Franka Emika Panda (modified DH, FER limits) | Haddadin et al., IEEE Robot. Autom. Mag. 29(2) 2022; Franka robot specification; franka_description URDF | 10.1109/MRA.2021.3138382 | P | [problems_robot](spec/problems_robot.md) |

## Models (`make_model(key, **kw)`)

| key | description | source | DOI | check | spec |
|---|---|---|---|---|---|
| `znn` | design-formula ZNN, dE/dt = -gamma g(t) Φ(E); any activation and gain schedule; resolvent form under implicit schemes | Zhang, Jiang, Wang, IEEE TNN 13(5) 2002; Zhang & Ge, IEEE TNN 16(6) 2005, eq. (3) | 10.1109/TNN.2002.1031938; 10.1109/TNN.2005.857946 | P | DESIGN.md, [activations](spec/activations.md) |
| `gnn` | gradient neural network, x_dot = -gamma Jᵀ Φ(E) (no time-derivative information; O(1/gamma) lag) | Zhang, Chen, Tan, IEEE TAC 54(8) 2009 (lag analysis of the classical GNN) | 10.1109/TAC.2009.2023779 | T | DESIGN.md |
| `znn_consistent` | `znn` that honours the consistent error maps (PEB, exact flow, semi-implicit) and the error-memory look-ahead rate of the `consistent` schemes | ZNN-Bench construction on Polyakov, Efimov, Brogliato, SICON 57(1) 2019 and Jiménez-Rodríguez et al., IFAC-PapersOnLine 53(2) 2020 | 10.1137/18M1197345; 10.1016/j.ifacol.2020.12.806 | C | [consistent](spec/consistent.md) |
| `vpznn` | power-type varying-parameter ZNN, g(t) = ζ + t^p | Zhang et al., IEEE TCYB 48(11) 2018; Zhang & Zheng, IEEE TCYB 49(10) 2019; Zhang et al., IEEE/ASME TMECH 23(2) 2018 (formulas via the Biomimetics 10(5) 2025 review) | 10.1109/TCYB.2017.2760883; 10.1109/TCYB.2018.2841970; 10.1109/TMECH.2018.2799724; 10.3390/biomimetics10050279 | S | [variants](spec/variants.md) |
| `vpznn_exp` | exponential-type varying-parameter ZNN, g(t) = p^t + p | Zhang, Zheng, Wang, Neurocomputing 338 2019 (via the Biomimetics 2025 review) | 10.1016/j.neucom.2019.01.058 | S | [variants](spec/variants.md) |
| `ieznn` | integration-enhanced / noise-tolerant ZNN, dE/dt = -gamma E - λ ∫E | Jin, Zhang, Li, IEEE TNNLS 27(12) 2016; Jin, Zhang, Li, Zhang, IEEE TAC 62(2) 2017 (form restated in the DIEZNN paper, Eq. (6)) | 10.1109/TNNLS.2015.2497715; 10.1109/TAC.2016.2566880 | S | [variants](spec/variants.md) |
| `dieznn` | double-integral-enhanced ZNN, characteristic polynomial (s+1)(s+ξ)², rejects linear noise | Liao, Han, Cao, Li, Li, CAAI TIT 9(1) 2024, Eqs. (6)-(16) | 10.1049/cit2.12161 | P | [variants](spec/variants.md) |
| `hntznn` | harmonic-noise-tolerant ZNN, internal model of known frequencies, Li activation | Liao, Wang, Li, Guo, He, Front. Neurorobot. 16 2022, Eqs. (4)-(5), (11), (14) | 10.3389/fnbot.2022.928636 | P | [variants](spec/variants.md) |
| `sptc_nt_znn` | strictly predefined-time noise-tolerant ZNN, gain 1/(t_c - t) before t_c, sign-power term after | Li, Guo, Ma, Pan, IEEE TIE 71(1) 2024 (as transcribed in Yang et al. 2025, Table 1) | 10.1109/TIE.2023.3241393 | S | [variants](spec/variants.md) |
| `sptc_an_foznn` | conformable fractional-order strictly predefined-time anti-noise ZNN | Yang et al., Neural Netw. 186 2025, Eqs. (11)-(18) (arXiv:2503.01857) | 10.1016/j.neunet.2025.107279 | P | [variants](spec/variants.md) |
| `gznn` | gradient-ZNN hybrid: ZNN design on the gradient error JᵀE, with Eq. (39) regularisation when JᵀJ is singular | Stanimirović, Mourtas, Katsikis, Kazakovtsev, Krutikov, Mathematics 10(22) 2022, Eqs. (31)-(40) | 10.3390/math10224292 | P | [variants](spec/variants.md) |
| `znn_explicit_inverse` | explicit inversion ZNN X_dot = -X A_dot X - gamma X Φ(AX - I) (no linear solve); requires `matrix_inverse`; locally convergent, so start from `newton_schulz_init` | Getz & Marsden, Proc. 34th IEEE CDC 1995, Thm 2.1, Eq. (1) (linear Φ); Φ-activated form: Guo & Zhang, Neurocomputing 2012 (not accessible) | 10.1109/CDC.1995.478664; 10.1016/j.neucom.2012.05.012 | P (linear) / U (Φ) | [structured_baselines](spec/structured_baselines.md) |
| `znn_explicit_inverse_right` | mirror form for E = XA - I; requires `matrix_inverse_right` | mirror substitution of the above (ASSUMPTION) | as above | U | [structured_baselines](spec/structured_baselines.md) |
| `davidenko` | Davidenko continuation flow x_dot = -J⁻¹ ∂E/∂t (no error feedback: keeps E(x0, 0)) | Davidenko, Dokl. Akad. Nauk SSSR 88 1953 (no DOI); Simonetto et al., IEEE TSP 64(17) 2016, Eq. (6); Allgower & Georg 2003 | 10.1109/TSP.2016.2568161; 10.1137/1.9780898719154 | P | [structured_baselines](spec/structured_baselines.md) |
| `newton_flow` | continuous Newton method on the frozen problem, x_dot = -gamma J⁻¹ E (ZNN without time-derivative information) | Gavurin, Izv. Vyssh. Uchebn. Zaved. Mat. 1958 (no DOI); Smale, J. Math. Econ. 3 1976 | 10.1016/0304-4068(76)90019-7 | T | [structured_baselines](spec/structured_baselines.md) |

| `lcc_znn` | low-computational-complexity ZNN: ZNN design on the scalar energy ν = ‖f‖²/2, x_dot = -Jᵀf (fᵀf_t + μφ(ν))/‖Jᵀf‖², activation (13) φ(z) = sig^r(z)e^{|z|}; one Jᵀv product, no solve | Zheng, Li, Zhang, IEEE TAC 69(7) 2024, Eqs. (10)-(13) (accepted manuscript, Oulu repository via Wayback) | 10.1109/TAC.2023.3319132 | P | [structured_extra](spec/structured_extra.md) |
| `hgnn_znn` | hybrid GNN-ZNN for matrix inversion, A X_dot = -γ(AAᵀ + I)F(AX - I) (constant A; `feedforward=True` adds the ZNN derivative term, an extension) | Dai, Chen, Xiao, Jia, He, IEEE TII 18(4) 2022, Eq. (10) | 10.1109/TII.2021.3093115 | P | [structured_extra](spec/structured_extra.md) |
| `fa_gnn_znn` | fuzzy-adaptive GNN-ZNN, gain γ + ν(‖E‖_F) from a Mamdani controller (trapezoidal memberships, max-min, middle-of-maximum) | Dai et al. 2022, Eqs. (2)-(4), (11) | 10.1109/TII.2021.3093115 | P (structure) / ASSUMPTION (membership breakpoints, not printed) | [structured_extra](spec/structured_extra.md) |

**Not registered** (primary text not obtainable; the design contract forbids registering a model from a secondary restatement alone): `inverse_free_znn`, the inverse-free ZNN of Chen, Pan, Zhang, Li, Tan, Neural Netw. 178 (2024) 106462, doi:10.1016/j.neunet.2024.106462 (ScienceDirect bot wall; the Oulu accepted manuscript is behind a Cloudflare check; no archived copy). Its secondary restatement (Xiao et al., IEEE TAI 2025, doi:10.1109/TAI.2025.3570287, Eq. (10)) equals `make_model("lcc_znn", activation="linear")`; results obtained that way must be labelled "LCCZNN (12) with linear φ", not "Chen et al. 2024". See `models/structured_extra.py::NOT_REGISTERED`.

### Gain schedules (`models/gains.py::SCHEDULES`; pass as `gain=` to `znn`)

| key | g(t) | source | DOI | check |
|---|---|---|---|---|
| `power_vp` | ζ + t^p (default t^p + p) | VP-ZNN papers above (Biomimetics 2025 review, Table 2) | 10.3390/biomimetics10050279 | S |
| `exp_vp` | p^t + p | Zhang, Zheng, Wang, Neurocomputing 338 2019 | 10.1016/j.neucom.2019.01.058 | S |
| `eps_exp` | ε e^t | Zhang, Zheng, Li, Deng, Xiao, Huang, Neurocomputing 319 2018 | 10.1016/j.neucom.2018.07.005 | S |
| `arccot_exp` | β exp(λ1 arccot t + λ2 t) | Xiao & He, IEEE TII 17(11) 2021 | 10.1109/TII.2021.3058343 | S |
| `predefined_time` | 1/(t_c - t) before t_c, `after` afterwards | SPTC-NT-ZNN (Li, Guo, Ma, Pan 2024) | 10.1109/TIE.2023.3241393 | S |
| `conformable` | max(t, t_floor)^(α-1) | Yang et al., Neural Netw. 186 2025, Eqs. (11)-(12) | 10.1016/j.neunet.2025.107279 | P |

## Activations (`znnbench.activations.get(key, **kw)`; pass as `activation=` to a model)

| key | Φ(e) (element-wise) | class | source | DOI | check |
|---|---|---|---|---|---|
| `linear` | e | exponential | Zhang, Jiang, Wang 2002; Zhang & Ge 2005, act. (i) | 10.1109/TNN.2002.1031938 | S |
| `power` | e^p, odd p ≥ 3 | asymptotic | Zhang, Jiang, Wang 2002; Zhang & Ge 2005 | 10.1109/TNN.2002.1031938; 10.1109/TNN.2005.857946 | S |
| `bipolar_sigmoid` | (1 - e^{-ξe})/(1 + e^{-ξe}) | exponential | Zhang & Ge 2005, act. (iii) | 10.1109/TNN.2005.857946 | S |
| `power_sigmoid` | e^p for \|e\| ≥ 1, scaled bipolar sigmoid otherwise | exponential | Zhang & Ge 2005, act. (iv) | 10.1109/TNN.2005.857946 | S |
| `power_sum` | Σ_{k=1}^{N} e^{2k-1} | exponential | Yang & Zhang, Neural Comput. Appl. 22 2013 | 10.1007/s00521-011-0692-5 | U |
| `hyperbolic_sine` | sinh(ξe) | exponential | Zhang, Jin, Ke, ComSIS 9(4) 2012 | 10.2298/CSIS120121043Z | P |
| `sign` | sign(e) | finite | Zhang, Ding, Qiu, Zhang, Li, Inf. Process. Lett. 124 2017 | 10.1016/j.ipl.2017.04.008 | U |
| `sign_power` | sign(e)\|e\|^r, 0 < r < 1 | finite | Zhang, Chen, Li et al., CAAI TIT 6(4) 2021, eqs. (8), (11) | 10.1049/cit2.12019 | P (not re-verified) |
| `saturated_sign_power` | sign(e) min(\|e\|^r, s) (SZNN) | finite | Zhang, Chen, Li et al., CAAI TIT 2021, Sec. 3.4 | 10.1049/cit2.12019 | P (not re-verified) |
| `sign_bi_power` | ½(sig^r(e) + sig^{1/r}(e)) | fixed (derived; source claims finite) | Li, Chen, Liu, Neural Process. Lett. 37 2013 | 10.1007/s11063-012-9241-1 | S |
| `sfaf` | b1 sig^r(e) + b2 e | finite | Xiao, Zhang, Li, IEEE TSMC 49(11) 2019; Hu et al. 2020 | 10.1109/TSMC.2018.2836968; 10.1016/j.neucom.2020.08.037 | U |
| `wsbp` | k1 sig^k(e) + k2 sig^{1/k}(e) + k3 e | fixed (derived; source claims finite) | Xiao, Neurocomputing 173 2016; Chai et al., IJCIS 13(1) 2020, eq. (6) | 10.1016/j.neucom.2015.08.031; 10.2991/ijcis.d.200527.001 | S |
| `vaf` | (k1\|e\|^p + k2\|e\|^q) sign(e) + k3 e + k4 sign(e) | predefined | Xiao et al., Neural Netw. 117 2019; Zhang & Yin, AIMS Math. 7(2) 2022, eq. (4.1) | 10.1016/j.neunet.2019.05.005; 10.3934/math.2022129 | S |
| `fixed_time` | (b1\|e\|^κ + b2\|e\|^η) sign(e) + b3 e | fixed | Jin, Zhu, Zhao, Chen, Appl. Soft Comput. 130 2022 (via the Biomimetics 2025 review, Table 1) | 10.1016/j.asoc.2022.109691 | S (weak) |
| `pvaf1` | power-versatile activation 1 | predefined | Zhou & Tan, Bull. Pol. Acad. Sci. Tech. Sci. 70(3) 2022, eq. (6) | 10.24425/bpasts.2022.141307 | P |
| `pvaf2` | power-versatile activation 2 | predefined | Zhou & Tan 2022, eq. (7), bound (15) | 10.24425/bpasts.2022.141307 | P |
| `predefined_exp` | γ1 exp(\|e\|^p)\|e\|^{1-p} sign(e)/p + γ2 e + γ3 sign(e) | predefined | Li, Liao, Xiao, Lu, Neurocomputing 337 2019 (F2) | 10.1016/j.neucom.2019.01.072 | S |
| `strict_predefined` | e/(t_c - t) before t_c; e + sig^p(e) + ξ sign(e) after (time-varying) | strictly predefined | Li, Guo, Ma, Pan, IEEE TIE 71(1) 2024 (via Yang et al. 2025, Table 1) | 10.1109/TIE.2023.3241393 | S |

These activations are not in the shared registry and are resolved locally:

| key | where | Φ(e) | source | DOI | check |
|---|---|---|---|---|---|
| `signed_power_sum` | `discretizations/consistent.py` (`znn_consistent`) | Σ a_i sign(e)\|e\|^{p_i} | generic container for sign, sign-bi-power and the bi-power fixed-time field (Polyakov 2012; Aldana-López et al. 2019) | 10.1109/TAC.2011.2179869; 10.1002/rnc.4600 | C |
| `jr_predefined` | `discretizations/consistent.py` (`znn_consistent`) | predefined-time field of JR20, Eq. (1) | Jiménez-Rodríguez et al., IFAC-PapersOnLine 53(2) 2020 | 10.1016/j.ifacol.2020.12.806 | P |
| `li` | `models/variants.py` (`hntznn`, `activation="li"`) | sig^r(e) + sig^{1/r}(e) (twice `sign_bi_power`) | Liao, Wang, Li, Guo, He, Front. Neurorobot. 16 2022, Eqs. (4)-(5) | 10.3389/fnbot.2022.928636 | P |

## Discretizations (`make_discretization(key, **kw)`)

`order` is `expected_order`, the slope p in MSSRE = O(tau^p) for linear ZNN at fixed h. "causal" means the scheme samples data only at t_k.

| key | description | causal | order | source | DOI | check | spec |
|---|---|---|---|---|---|---|---|
| `euler` | forward Euler look-ahead (x_{k+1} - x_k) | yes | 2 | classical; Euler-type DTZNN | — | T | DESIGN.md |
| `taylor_zhang` | 4-instant Taylor-Zhang formula (2, -3, 2, -1)/2 | yes | 3 | Zhang, Jin, Guo, Yin, Chou, JCAM 273 2015; step range: Zhang et al., IEEE TNNLS 30(3) 2019 | 10.1016/j.cam.2014.05.027; 10.1109/TNNLS.2018.2861404 | S | [discretizations](spec/discretizations.md) |
| `zead_4i_general` | 4-instant ZeaD family, parameter a1 < 0 (default -1/4) | yes | 3 | Hu, Kang, Zhang, Neurocomputing 306 2018 (via Sun, Tian, Wang 2019, Table 1) | 10.1016/j.neucom.2018.03.053; 10.1155/2019/4745759 | S | [discretizations](spec/discretizations.md) |
| `zead_4i_6321` | 4-instant (6, -3, -2, -1)/10, h in (0, 0.8) | yes | 3 | Guo, Lin, Su, Sun, Huang, Numer. Algor. 77 2018; range: Zhang, Qi, Li, Qiu, Yang, Numer. Algor. 81 2019 | 10.1007/s11075-017-0302-4; 10.1007/s11075-018-0561-8 | S | [discretizations](spec/discretizations.md) |
| `zead_4i_fifd` | 4-instant (5, -3, -1, -1)/8 | yes | 3 | Li, Mao, Uhlig, Zhang, JCAM 327 2018 (via Uhlig 2019, item (D)) | 10.1016/j.cam.2017.06.017 | S | [discretizations](spec/discretizations.md) |
| `zead_5i` | 5-instant (8, 1, -6, -5, 2)/18 | yes | 4 | Li, Mao, Uhlig, Zhang, Numer. Algor. 81 2019 (via Uhlig 2019 (E), Uhlig 2024 eq. (4i)) | 10.1007/s11075-018-0564-5 | S | [discretizations](spec/discretizations.md) |
| `zead_5i_general` | 5-instant family, a1 in (1/12, 1/6) (default 1/9 = `zead_5i`) | yes | 4 | Zhang, He, Hu, Guo, Li, Shi, JCAM 347 2019 (via Sun-Tian-Wang, Table 1) | 10.1016/j.cam.2018.08.017 | S | [discretizations](spec/discretizations.md) |
| `zead_5i_mahuang2025` | 5-instant (36, -1, -18, -27, 10)/78 (= `zead_5i_general` a1 = 5/39); Ma-Huang DTZNN Eq. (23); h_ours = 6/13 h_paper | yes | 4 | Ma & Huang, Front. Neurorobot. 19 2025, Thm 1, Eqs. (14), (20) | 10.3389/fnbot.2025.1576473 | P | [inverse_free_dtznn](spec/inverse_free_dtznn.md) |
| `zead_6i_guo2017` | 6-instant (24, -5, -12, -6, -4, 3)/48 | yes | 4 | Guo, Nie, Yan, IEEE TSMC-S 47(8) 2017, Thm 4, eq. (6) | 10.1109/TSMC.2017.2656941 | U (exact moments) | [discretizations](spec/discretizations.md) |
| `zead_6i_qiu2019` | 6-instant (13, -6, -2, -4, -3, 2)/24 | yes | 4 | Qiu, Zhang, Guo, Yang, Li, Numer. Algor. 81 2019, eqs. (16), (18) (via Uhlig 2019 (F)) | 10.1007/s11075-018-0581-4 | S | [discretizations](spec/discretizations.md) |
| `zead_7i_general` | 7-instant family eq. (8), parameters (p1, p2), default (-1, -1) near the domain edge | yes | 5 | Yang, Zhang, Hu, Qiu, IEEE TNNLS 31(9) 2020 | 10.1109/TNNLS.2019.2938866 | U (domain (16) reproduced exactly) | [discretizations](spec/discretizations.md) |
| `zead_7i_cang2024` | 7-instant (110, 42, -89, -102, 12, 44, -17)/276 (= `zead_7i_general` p1 = -89/110, p2 = -51/55); Taylor DTZNN-III; h_ours = 55/138 h_paper | yes | 5 | Cang et al., Appl. Soft Comput. 157 2024 (as printed in Ma & Huang 2025, Eq. (10)) | 10.1016/j.asoc.2024.111511; 10.3389/fnbot.2025.1576473 | S | [inverse_free_dtznn](spec/inverse_free_dtznn.md) |
| `zead_8i_wu2020` | 8-instant, eq. (12) | yes | 5 | Wu, Zhang, Guo, Li, Ming, IEEE Access 8 2020 | 10.1109/ACCESS.2020.3045780 | U (exact moments) | [discretizations](spec/discretizations.md) |
| `zead_8i_chen2019` | 8-instant (η1 = η3 = 0, η2 = -9/10) | yes | 5 | Chen & Zhang, IEEE Access 7 2019; as given in Chen, Kang, Zhang, Mathematics 11 2023, eq. (7) | 10.1109/ACCESS.2019.2938840; 10.3390/math11153354 | U (exact moments) | [discretizations](spec/discretizations.md) |
| `zead_10i_wu2020` | 10-instant, Table 1 rows 1-6 (`row=1` default) | yes | 6 | Wu et al., IEEE Access 8 2020, eq. (14) | 10.1109/ACCESS.2020.3045780 | U (exact moments) | [discretizations](spec/discretizations.md) |
| `zead_11i_chen2023` | 11-instant, Thm 3 eq. (6) | yes | 6 | Chen, Kang, Zhang, Mathematics 11 2023 | 10.3390/math11153354 | U (exact moments) | [discretizations](spec/discretizations.md) |
| `uhlig_2_3` | Uhlig 5-instant formula of type 2_3 (same coefficients as `zead_5i`) | yes | 4 | Uhlig, J. Difference Equ. Appl. 25 2019, item (E); Uhlig, Numer. Math. 156 2024, eq. (4i) | 10.1080/10236198.2019.1627343; 10.1007/s00211-023-01393-5 | P | [discretizations](spec/discretizations.md) |
| `uhlig_3_4` | Uhlig 7-instant formula ("3_3" of [U19] = type 3_4 of [U24]), seed [a, 110, -40] (h_max = 0.054 at a = 1) | yes | 5 | Uhlig, JDEA 25 2019, p. 7 | 10.1080/10236198.2019.1627343 | P | [discretizations](spec/discretizations.md) |
| `uhlig_4_6` | Uhlig formula "4_5a" of [UF] = type 4_6 of [U24] (10 instants, non-rational 16-digit coefficients) | yes | 6 | Uhlig, Linear Multilinear Algebra 68 2020 (arXiv:1904.10568), p. 9 | 10.1080/03081087.2019.1648375 | P | [discretizations](spec/discretizations.md) |
| `ab2` / `ab3` / `ab4` | Adams-Bashforth 2/3/4 on the ZNN vector field | yes | 3 / 4 / 5 | Hairer, Nørsett, Wanner, Solving ODEs I, Sec. III.1 | — | T | [discretizations](spec/discretizations.md) |
| `rk4` | classical Runge-Kutta 4 (samples data at t_k + tau/2 and t_k + tau); `expected_order=None` means residual order **2 at fixed h = γτ** (stiff order reduction: linear ZNN is a Prothero-Robinson problem and RK4 has stage order 1) and **4 at fixed γ** | no | 2 (fixed h) / 4 (fixed γ) | classical; Prothero & Robinson, Math. Comp. 28 1974; Biswas et al., SINUM 63(4) 2025 | 10.1090/S0025-5718-1974-0331793-2; 10.1137/23M1606812 | T | [rk4_order_reduction](spec/rk4_order_reduction.md) |
| `implicit_euler_err` | Euler with the error dynamics stepped by the activation resolvent (implicit) | yes | 2 | backward Euler of the error dynamics, cf. Polyakov-Efimov-Brogliato 2019 | 10.1137/18M1197345 | C | DESIGN.md |
| `consistent` | parametric factory `consistent(formula, error_map, memory)` of the schemes below | yes | 2/3 | as below | — | C | [consistent](spec/consistent.md) |
| `implicit_taylor_zhang` | Taylor-Zhang look-ahead with the resolvent error map and error-memory rate (novel) | yes | 3 | Acary & Brogliato, Syst. Control Lett. 59(5) 2010 (implicit principle); Huber, Acary, Brogliato, IEEE TAC 61(10) 2016 | 10.1016/j.sysconle.2010.03.002; 10.1109/TAC.2015.2506991 | C | [consistent](spec/consistent.md) |
| `implicit_taylor_zhang_naive` | the same without error memory (negative control) | yes | 3 | as above | as above | C | [consistent](spec/consistent.md) |
| `peb_euler` / `peb_taylor_zhang` | homogeneity-based consistent map of PEB19 (Thms 4.1/5.1, Lie splitting) | yes | 2 / 3 | Polyakov, Efimov, Brogliato, SICON 57(1) 2019 | 10.1137/18M1197345 | P | [consistent](spec/consistent.md) |
| `exact_flow_euler` / `exact_flow_taylor_zhang` | exact one-step flow of e' = -Φ(e) in settling-time coordinates | yes | 2 / 3 | Jiménez-Rodríguez et al., IFAC-PapersOnLine 53(2) 2020, Eq. (7); Aldana-López et al., IJRNC 29(12) 2019 | 10.1016/j.ifacol.2020.12.806; 10.1002/rnc.4600 | P | [consistent](spec/consistent.md) |
| `semi_implicit_euler` / `semi_implicit_taylor_zhang` | frozen-coefficient linearly implicit map | yes | 2 / 3 | PEB19, Sec. 1.2, Eq. (5.3) | 10.1137/18M1197345 | P | [consistent](spec/consistent.md) |
| `pc_newton` | prediction-correction, Newton correction with t_{k+1} data (NTT) | no | — | Simonetto, Mokhtari, Koppel, Leus, Ribeiro, IEEE TSP 64(17) 2016, Alg. 2 | 10.1109/TSP.2016.2568161 | P | [structured_baselines](spec/structured_baselines.md) |
| `pc_gradient` | prediction-correction, gradient correction (GTT) | no | 2 | Simonetto et al. 2016, Alg. 1, Eq. (8) | 10.1109/TSP.2016.2568161 | P | [structured_baselines](spec/structured_baselines.md) |
| `broyden_dtznn_zhang2012` | Euler DTZNN with J⁻¹ replaced by the Broyden secant matrix (inverse-free, O(n²) per step); `secant='current'` (printed) or `'previous'`; the printed Eqs. (6)-(7) do not reproduce the source's Table 1 row | yes | 2 (claimed; holds on guo2018 only) | Ma & Huang 2025, Eqs. (6)-(7); Zhang, Peng, Li, Shi, Ling, ICCECT 2012 (not accessible) | 10.3389/fnbot.2025.1576473 | S | [inverse_free_dtznn](spec/inverse_free_dtznn.md) |
| `cdtznn_mahuang2025` | combined DTZNN: 5-instant x-recursion with M_k ≈ J⁻¹ from a ZNN inversion recursion (no linear solve); square J only; default M_0 = Newton-Schulz (flagged; the paper's figures suggest M_0 = I) | yes | 4 | Ma & Huang 2025, Eqs. (24)-(26), (36)-(38) | 10.3389/fnbot.2025.1576473 | P | [inverse_free_dtznn](spec/inverse_free_dtznn.md) |
| `newton_tracking` | correction-only Newton tracking | no | — | Simonetto et al. 2016 | 10.1109/TSP.2016.2568161 | P | [structured_baselines](spec/structured_baselines.md) |
| `running_gradient` | correction-only gradient ("AGT without prediction") | no | 1 | Simonetto et al. 2016, Sec. IV-B; Popkov, Autom. Remote Control 66(6) 2005 | 10.1109/TSP.2016.2568161 | P | [structured_baselines](spec/structured_baselines.md) |
| `pc_newton_causal` / `pc_gradient_causal` / `newton_tracking_causal` | causal variants: correct with t_k data, then predict | yes | 2 / 2 / 1 | ZNN-Bench construction on Simonetto et al. 2016 | 10.1109/TSP.2016.2568161 | C | [structured_baselines](spec/structured_baselines.md) |

[U19] = Uhlig, JDEA 2019; [UF] = Uhlig, LAMA 2020; [U24] = Uhlig, Numer. Math. 2024. Keys follow the [U24] labels, where j_s means j + s instants (see docs/spec/discretizations.md). Not implemented: Uhlig's order-7/8 formulas (5_5, 5_6b and any order-8 type). Their coefficients exist only in his MATLAB files (`Polyksrestcoeff3.m`), which are no longer online and were not captured by the Wayback Machine; none of his papers prints them. Every source tried is listed in [uhlig_high](spec/uhlig_high.md), and `discretizations/uhlig_high.py` registers nothing (it holds the drop-in path for a future primary copy).

## Noises (`make_noise(key, **kw)`; added in the design-formula space, J x_dot = model + ν)

| key | ν(t) (every entry) | source | DOI | check | spec |
|---|---|---|---|---|---|
| `constant` | level | Jin, Zhang, Li, IEEE TNNLS 27(12) 2016 (noise classes); values [8]: Yan et al., IEEE Access 7 2019 | 10.1109/TNNLS.2015.2497715; 10.1109/ACCESS.2019.2907746 | P (values) | [noise_info](spec/noise_info.md) |
| `linear` | offset + slope·t | as `constant`; values [8t]: Yan et al. 2019 | 10.1109/ACCESS.2019.2907746 | P (values) | [noise_info](spec/noise_info.md) |
| `gaussian` | std·N(0, 1), drawn at every evaluation | benchmark white noise | — | C | DESIGN.md |
| `harmonic` | amp·sin(ωt + φ) | Liao, Wang, Li, Guo, He, Front. Neurorobot. 16 2022, eq. (7) | 10.3389/fnbot.2022.928636 | P | [noise_info](spec/noise_info.md) |
| `uniform` | U[low, high] (bounded random) | Liao, Hua, Cao, Katsikis, Li, Mathematics 10(15) 2022, Remark 3; Yan et al. 2019 ([7.5, 8.5]) | 10.3390/math10152817; 10.1109/ACCESS.2019.2907746 | P | [noise_info](spec/noise_info.md) |
| `polynomial` | Σ c_k t^k | Liao et al., Mathematics 2022, Sec. 3 | 10.3390/math10152817 | P | [noise_info](spec/noise_info.md) |
| `quadratic` | a t² + b t + c | Liao et al., Mathematics 2022, Sec. 3, Thm 3 | 10.3390/math10152817 | P | [noise_info](spec/noise_info.md) |
| `multi_harmonic` | Σ Γ_k sin(2π f_k t + φ_k) | Liao et al., Front. Neurorobot. 2022, eq. (12), Sec. 5 | 10.3389/fnbot.2022.928636 | P | [noise_info](spec/noise_info.md) |
| `combined` | constant + ramp + harmonic (+ bounded random) | ZNN-Bench composite; closest: Yang, Wang, Voyles, Ma, arXiv:2412.20477 (sin t + cos t + bounded noise) | 10.26599/TST.2024.9010202 | C | [noise_info](spec/noise_info.md) |
| `random_walk` | Wiener-type ν(t') = ν(t) + σ sqrt(t' - t) ξ | benchmark extension (no ZNN source found) | — | C | [noise_info](spec/noise_info.md) |
| `impulse` | rectangular bursts, periodic or Poisson | benchmark extension (no ZNN source found) | — | C | [noise_info](spec/noise_info.md) |

Not implemented: state-dependent ("vanishing") noise, |ν| ≤ δ|e| (Zhao et al., AIMS Math. 10(6) 2025, doi:10.3934/math.2025628; Jin et al., Front. Neurorobot. 16 2022, doi:10.3389/fnbot.2022.1065256). It needs E in the noise call (CORE_CHANGE_REQUESTS.md, noise_info item 1).

## Wrappers (runner `wrappers:` key = class name in `znnbench.wrappers`; the target is inferred from the base class)

| key | target | description | source | DOI | check | spec |
|---|---|---|---|---|---|---|
| `QuantizedDiscretizer` | disc | b-bit signed fixed-point state (and optionally derivative) with saturation | rounding-noise model of Widrow & Kollár, Quantization Noise, 2008 | 10.1017/CBO9780511754661 | C | [noise_info](spec/noise_info.md) |
| `JitteredProblem` | problem | data sampled at t_k + δ_k (i.i.d., seeded) while the model believes t_k | benchmark construction | — | C | [noise_info](spec/noise_info.md) |
| `DelayedDataProblem` | problem | data arrive with a fixed delay | benchmark construction | — | C | [noise_info](spec/noise_info.md) |
| `EstimatedDerivativeProblem` | problem | causal estimate of ∂E/∂t from past samples (`savgol`, `alpha_beta`, `kalman`) | Savitzky & Golay 1964; Benedict & Bordner 1962; Kalata 1984; Bar-Shalom, Li, Kirubarajan 2001 | 10.1021/ac60214a047; 10.1109/TAC.1962.1105477; 10.1109/TAES.1984.310438; 10.1002/0471221279 | T | [noise_info](spec/noise_info.md) |
| `CountingProblem` | problem | call counts and wall-clock per step (with `dense_flops`, `profile_step_cost`) | benchmark construction | — | C | [noise_info](spec/noise_info.md) |

Also available through an import path (`"package.module:callable"`): `znnbench.problems.robot:StateRecorder`, a transparent discretizer wrapper that records the states needed for the joint-drift metric ([problems_robot](spec/problems_robot.md)). `runner.register_wrapper` is empty by default.
