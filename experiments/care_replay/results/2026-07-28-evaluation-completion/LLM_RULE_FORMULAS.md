# CARE 2.0 LLM Rule Formulas

This note records the equations executed by the current source-outcome router.
It is intentionally narrower than the prompt: the LLM proposes a bounded
`KernelSkillPatch`, while Python performs calibration, scoring, gating, and
candidate selection deterministically.

## 1. LLM output contract

For each source-target pair, the LLM returns a portfolio of patches

\[
P_k = (S_k, m_{k,j}, \beta_{k,0}, \beta_{k,T}, \lambda_k,
\tau_k, K_k, \gamma_k, c_k, r_k),
\]

where:

- \(S_k\) is a set of transfer scales;
- \(m_{k,j}\) is the multiplier for target role or descriptor \(j\);
- \(\beta_{k,0}\) and \(\beta_{k,T}\) define the exploration schedule;
- \(\lambda_k\) is source-prior strength;
- \(\tau_k\) and \(K_k\) are source-neighbor temperature and neighbor count;
- \(\gamma_k\) is source-interaction strength;
- \(c_k\) is LLM confidence;
- \(r_k\) contains calibration mode and minimum cross-validation gain.

The parser clips every field to the declared range. The LLM does not output a
candidate ID, target outcome, or final acquisition score.

Implementation: `scripts/run_llm_kernel_skill_evolution.py`,
`KernelSkillPatch` and `normalize_patches`.

## 2. Role-weighted GP kernel

For scale \(s \in S_k\), source role confidence \(c_j^{role}\), and LLM role
multiplier \(m_{k,j}\), the raw categorical weight is

\[
\widetilde w_{k,j}(s) = 1 + s\,c_j^{role}m_{k,j}.
\]

With normalization enabled,

\[
w_{k,j}(s) = \frac{\widetilde w_{k,j}(s)}
{J^{-1}\sum_{l=1}^{J}\widetilde w_{k,l}(s)}.
\]

The mixed kernel is

\[
K_{k,s}(x,x') = \exp\left(
-\frac{\lVert x_{num}-x'_{num}\rVert_2^2}{2\ell_n^2}
-\frac{\sum_j w_{k,j}(s)\,\mathbf{1}[x_j \ne x'_j]}{\ell_c}
\right).
\]

This is where a transferred skill changes the geometry of Bayesian
optimization: fields judged important by the source evidence and the LLM have
larger mismatch penalties.

Implementation: `scripts/run_transfer_weighted_kernel.py`,
`transfer_categorical_weights` and `mixed_kernel_weighted`.

## 3. Target-only acquisition anchor

For target observations available at round \(t\), the GP posterior gives
\(\mu_t(x)\) and \(\sigma_t(x)\). The two strong target-only acquisitions are

\[
UCB_t(x)=\mu_t(x)+\beta_t\sigma_t(x),
\]

\[
EI_t(x)=(\mu_t(x)-y_t^*-\xi)\Phi(z)+\sigma_t(x)\phi(z),
\quad z=\frac{\mu_t(x)-y_t^*-\xi}{\sigma_t(x)}.
\]

The target portfolio used by the router is

\[
a_{anchor,t}(x)=0.5\,R(UCB_t(x))+0.5\,R(EI_t(x)),
\]

where \(R\) maps candidate rank to \([0,1]\). The LLM beta schedule is linear:

\[
\beta_{k,t}=\beta_{k,0}+\frac{t}{T-1}
(\beta_{k,T}-\beta_{k,0}).
\]

Implementation: `scripts/run_llm_transfer_router.py`, `target_anchor_scores`,
and `scripts/run_transfer_weighted_kernel.py`, `scheduled_gp_beta`.

## 4. Source-outcome priors

### 4.1 Neighbor prior

Source outcomes are standardized:

\[
z_i^s=\frac{y_i^s-\bar y^s}{\max(sd(y^s),0.05)}.
\]

For mapped categorical roles, the normalized mismatch distance is

\[
d_{cat}(x_i^s,x)=
\frac{\sum_j \omega_j\,\delta_j(x_i^s,x)}{\sum_j\omega_j},
\quad
\omega_j=\max(0.02, w_j^{transfer}m_{k,j}),
\]

where \(\delta_j=0\) for a match, \(1\) for a mismatch, and \(0.5\) when one
side is missing. For MoleculeNet and Matbench descriptor families,

\[
d(x_i^s,x)=0.45d_{cat}+0.55d_{num};
\]

otherwise \(d=d_{cat}\). For the \(K_k\) nearest source observations,

\[
p_{neighbor,k}(x)=
\frac{\sum_{i\in N_{K_k}(x)}\exp(-d_i/\tau_k)z_i^s}
{\sum_{i\in N_{K_k}(x)}\exp(-d_i/\tau_k)}.
\]

### 4.2 Additive value prior

For mapped field-value pair \((j,v)\) with source support \(n_{jv}\),

\[
e_{jv}=\frac{n_{jv}}{n_{jv}+5}\;\operatorname{mean}
\{z_i^s:x_{i,j}^s=v\}.
\]

Candidate-level additive prior is the role-weighted mean of all matched
effects:

\[
p_{add,k}(x)=
\frac{\sum_{(j,v)\in M(x)}\omega_j e_{jv}}
{\sum_{(j,v)\in M(x)}\omega_j}.
\]

The router also constructs interaction priors from jointly supported mapped
roles. They use the same bounded calibration described below.

Implementation: `scripts/run_llm_transfer_router.py`, `aligned_source_prior`,
`source_additive_outcome_prior`, and `source_interaction_prior`.

## 5. Target calibration of a source prior

Given currently observed target pairs \((p_i,y_i)\), the router fits a
ridge-stabilized line

\[
\widehat y_i=b+ap_i,
\quad
a=\frac{\sum_i(p_i-\bar p)(y_i-\bar y)}
{\sum_i(p_i-\bar p)^2+0.05},
\quad b=\bar y-a\bar p.
\]

Leave-one-out gain compares this predictor with a train-mean baseline:

\[
g_{CV}=\operatorname{clip}_{[-2,1]}\left(
1-\frac{MSE_{LOO}(b+ap)}{MSE_{LOO}(\bar y_{train})}
\right).
\]

The prior is inactive when \(g_{CV}<g_{min}\). In `positive_only` mode it is
also inactive when \(a\le0\). Otherwise,

\[
q_{evidence}=\min(1,n_t/10),
\]

\[
q_{gain}=\min\left(1,
\frac{\max(0,g_{CV}-g_{min})}{\max(0.15,1-g_{min})}
\right),
\quad \rho=q_{evidence}\sqrt{q_{gain}},
\]

\[
\Delta_k(x)=\operatorname{clip}_{[-C_t,C_t]}
\left(\lambda_k\rho\,
\operatorname{clip}_{[-2.5s_y,2.5s_y]}(b+ap_k(x)-\bar y)\right),
\]

with

\[
C_t=\min(0.35,0.12+0.10\log(1+n_t)).
\]

This is the key evidence boundary: source outcomes propose a prior, but
revealed target observations determine its sign, reliability, and bounded
magnitude.

Implementation: `scripts/run_llm_transfer_router.py`, `linear_fit`,
`leave_one_out_gain`, and `calibrated_prior_adjustments`.

## 6. Expert activation and routing

For patch \(k\), kernel quality is measured by target leave-one-out MAE gain

\[
g_{kernel,k}=\operatorname{clip}_{[-2,1]}
\left(1-\frac{MAE_{patch,k}}{MAE_{target\ kernel}}\right).
\]

A prior expert is active when its calibrated CV gain is at least \(0.20\). A
kernel expert is active when \(g_{kernel,k}\ge0.05\) and its GP log marginal
likelihood is no worse than the target kernel. Let

\[
Q_k=\max(0,g_{CV,k},g_{interaction,k},g_{kernel,k}).
\]

The route score and expert weight are

\[
v_k=\log(\max(c_k,0.05))+3Q_k,
\quad
\pi_k=\frac{\exp(v_k)}{\sum_l\exp(v_l)}.
\]

The total transfer mass is

\[
M_t=\min(M_{max},0.55\max_k Q_k).
\]

After rank normalization, the CARE acquisition is

\[
a_{CARE,t}(x)=(1-M_t)a_{anchor,t}(x)
+M_t\sum_k\pi_k a_{k,t}(x).
\]

Weak evidence therefore converges continuously to the target-only anchor
rather than receiving a fixed minimum transfer weight.

Implementation: `scripts/run_llm_transfer_router.py`, lines implementing
`route_values`, `softmax_weights`, `transfer_mass`, and `combined_scores`.

## 7. Online and deployment gates

The router candidate is rejected online when any of the following holds:

- fewer than 10 target observations are available;
- maximum online quality is below 0.15;
- target-anchor acquisition loss exceeds
  \(B_t=\max(0.025,0.080\exp(-0.22t))\);
- transfer mass is above 0.45.

The outer protocol then uses 50 calibration seeds to freeze either the source
route or an exact matched-target fallback. No model selection is performed on
the 100 held-out seeds.

Implementation: `scripts/run_llm_transfer_router.py`, `router_gate_decision`,
and `scripts/run_calibrated_source_outcome_transfer.py`.

## 8. Metrics

For target budget \(T\), best-so-far and its area are

\[
b_t=\max_{i\le t} y_i,
\quad
AUC=\frac{1}{T}\sum_{t=1}^{T}b_t.
\]

Round saving is paired by seed:

\[
\Delta R=R_{baseline}-R_{CARE}.
\]

Positive \(\Delta R\) means CARE reaches the same quality threshold using fewer
target experiments. Missed thresholds are right-censored at \(T+1\).
