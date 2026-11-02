# QWFM for NMR Reconstruction

**Quantile–Wasserstein Flow Matching for NMR Relaxation Spectrum Reconstruction**

Generate multiple nonnegative, unit-mass relaxation spectra conditioned on a noisy NMR decay signal. This repository provides synthetic data generation, model training, evaluation, and visualization for a manuscript currently under review.

[Method](#method) · [Installation](#installation) · [Experiments](#experiments) · [Usage](#usage) · [Visualization](#visualization)

## Method

QWFM represents a spectrum by **quantile positions on the log-rate axis**. Interpolating quantiles transports spectral mass along that axis; histogram-linear flow matching (HFM) instead interpolates masses at fixed bins. Training uses simulated spectrum–decay pairs, with a Gamma-mixture target prior and an independent Dirichlet source. SMC posterior references are used only for evaluation.

### 1. Encode histograms as quantiles

For bin masses $`h_n`$ on normalized log-rate bins $`[b_{n-1},b_n)`$, $`b_n=n/N`$, assume uniform density within each bin. At midpoint mass levels $`u_\ell`$, invert the piecewise-linear CDF:

```math
\begin{aligned}
C_n&=\sum_{j=1}^{n}h_j,\quad C_0=0,\qquad u_\ell=\frac{\ell+1/2}{L},\\
q_\ell&=b_{n-1}+\frac{u_\ell-C_{n-1}}{h_n}(b_n-b_{n-1}),
\qquad C_{n-1}\le u_\ell\lt C_n.
\end{aligned}
```

Here $`\ell=0,\ldots,L-1`$; each $`q_\ell`$ is a spectral position at a fixed cumulative mass level.

### 2. Learn a conditional quantile flow

On this normalized axis, one-dimensional Wasserstein geometry gives
$`W_2^2=\int_0^1|Q_1(u)-Q_0(u)|^2\,du\approx L^{-1}\|\mathbf q_1-\mathbf q_0\|_2^2`$.
Linear interpolation of the full quantile functions is a $`W_2`$ geodesic. A conditional 1D U-Net learns its finite-coordinate velocity:

```math
\begin{aligned}
\mathbf q_t&=(1-t)\mathbf q_0+t\mathbf q_1,\qquad t\sim\mathcal U(0,1),\\
\mathcal L_{\mathrm{QWFM}}&=
\mathbb E\!\left[\frac{1}{L}
\left\|v_\theta(\mathbf q_t,t,\mathbf c)-(\mathbf q_1-\mathbf q_0)\right\|_2^2\right].
\end{aligned}
```

The condition $`\mathbf c`$ contains the observed decay, echo times, and noise level.

### 3. Generate conditional samples

At test time, fix $`\mathbf c`$, draw a new source histogram $`\widehat{\mathbf h}_0\sim\mathrm{Dirichlet}(\mathbf 1_N)`$, and encode it as $`\widehat{\mathbf q}_0`$ using step 1. The learned velocity defines the sampling ODE and its integral form:

```math
\begin{aligned}
\frac{\mathrm d\widehat{\mathbf q}_t}{\mathrm dt}
&=v_\theta(\widehat{\mathbf q}_t,t,\mathbf c),\qquad t\in[0,1],\\
\widehat{\mathbf q}_1
&=\widehat{\mathbf q}_0+\int_0^1v_\theta(\widehat{\mathbf q}_t,t,\mathbf c)\,\mathrm dt.
\end{aligned}
```

In practice, start from $`\mathbf q^{(0)}=\widehat{\mathbf q}_0`$ and take $`K=\mathrm{NFE}`$ explicit Euler steps, with $`\Delta t=1/K`$ and $`t_k=k\Delta t`$. Each step clips positions before applying isotonic projection:

```math
\mathbf q^{(k+1)}=
\Pi_{\uparrow}\!\left(
\mathrm{clip}_{[0,1]}\!\left[
\mathbf q^{(k)}+\Delta t\,v_\theta(\mathbf q^{(k)},t_k,\mathbf c)
\right]\right),\qquad k=0,\ldots,K-1.
```

Here $`\Pi_{\uparrow}`$ is the equal-weight L2 isotonic projection onto nondecreasing vectors. Decode the final positions $`\widehat q_\ell=[\mathbf q^{(K)}]_\ell`$ below. Independent source draws give multiple reconstructions for the same fixed condition, without requiring a target spectrum.

### 4. Decode quantiles into bin masses

The **linear-CDF decoder** interpolates generated quantiles and recovers cumulative mass at bin boundaries:

```math
\begin{aligned}
\widehat Q(u)&=(1-\alpha)\widehat q_\ell+\alpha\widehat q_{\ell+1},
\qquad \alpha=\frac{u-u_\ell}{u_{\ell+1}-u_\ell},\quad u_\ell\le u\le u_{\ell+1},\\
\widehat F_n&=\int_0^1\mathbf 1\{\widehat Q(u)\lt b_n\}\,du,\qquad 1\le n\lt N,\\
\widehat h_n&=\widehat F_n-\widehat F_{n-1},\qquad n=1,\ldots,N.
\end{aligned}
```

Endpoint quantiles use half-step extrapolation clipped to $`[0,1]`$, with $`\widehat F_0=0`$ and $`\widehat F_N=1`$. See the [representation utilities](codes/utils/nmr_utils.py), [training objective](codes/trainers/quantile_wasserstein_trainer.py), and [sampler](codes/samplers/quantile_wasserstein_sampler.py).

## Installation

Use **Linux/Bash, Python 3.11+, and CUDA-compatible PyTorch**. Install the appropriate PyTorch build first, then:

```bash
git clone https://github.com/rechido/qwfm_nmr_reconstruction.git
cd qwfm_nmr_reconstruction
python -m pip install -r requirements.txt
```

Dependencies are pinned or bounded in [requirements.txt](requirements.txt). The workflows generate synthetic 1D NMR data locally; no dataset download is required. Pretrained checkpoints are not included.

## Experiments

| Setting | Measurements M | Spectrum bins N | QWFM quantiles L | Complete command list |
| --- | ---: | ---: | ---: | --- |
| `m32n128` | 32 | 128 | 128 | [Commands](run_examples_m32n128.txt) |
| `m64n128` | 64 | 128 | 128 | [Commands](run_examples_m64n128.txt) |
| `m128n256` | 128 | 256 | 256 | [Commands](run_examples_m128n256.txt) |

Each setting includes **HFM, QWFM, MSE+W1 regression, and CVAE (z=16)**. HFM/QWFM sweep NFE (network evaluations) over **1, 4, 16, 64, 256**; regression and CVAE are NFE-independent. QWFM uses **L=N and linear-CDF decoding**.

| Evaluation | Included experiments |
| --- | --- |
| Precision/recall and F1 summary | Comparison against SMC references under histogram L2 and log-rate W1/W2 |
| Energy score / kernel score (ES/KS) | Independent-test evaluation under L2 and exact log-rate W2 |
| Forward consistency | Decay errors for individual spectra and the mean spectrum |
| Qualitative reconstruction | Five truth-selected cases: a broad single peak and two to five separated peaks; spectra and decays at flow NFE=256 |

PR uses 32 conditions, 256 predictions per condition, and three audited 4,096-particle SMC banks; its W2 distance uses 2,048 midpoint quantiles. ES/KS and forward consistency use 16,384 independent test pairs, with eight draws per stochastic model and one prediction for regression.

## Usage

**Copy commands from the lists above; no shell arguments are required.** All settings are inside [the scripts](scripts/paper/). Run from the repository root in your activated environment. Follow the list order, not the numerical order of filenames.

Run shared preparation once per setting, then complete any model's **training → generation → evaluation → visualization** block. The defaults use **four GPUs for training/independent sampling and one for SMC/PR**. Set `GPU_IDS` and `NUM_PROCESSES` inside the scripts for your machine; when changing GPU count, adjust batch size and gradient accumulation to retain the effective batch of **2,048**. Full runs train for **200,000 updates**, and evaluation requires the final EMA checkpoint.

Example: complete QWFM at M64/N128 without running any other model:

```bash
# Shared preparation; the SMC audit must complete without rejection.
bash scripts/paper/m64n128/00_prepare_test_conditions.sh
bash scripts/paper/m64n128/10_build_reference_run0.sh
bash scripts/paper/m64n128/11_build_reference_run1.sh
bash scripts/paper/m64n128/12_build_reference_run2.sh
bash scripts/paper/m64n128/13_audit_references.sh

# QWFM: train, save predictions, evaluate, then visualize.
bash scripts/paper/m64n128/02_train_quantile_wasserstein.sh
bash scripts/paper/m64n128/31_generate_quantile_wasserstein_samples.sh
bash scripts/paper/m64n128/41_evaluate_quantile_wasserstein_scores.sh
bash scripts/paper/m64n128/51_evaluate_quantile_wasserstein_forward.sh
bash scripts/paper/m64n128/21_evaluate_pr_quantile_wasserstein.sh
bash scripts/paper/m64n128/61_visualize_quantile_wasserstein.sh
```

PR generates its own predictions for the SMC conditions. ES/KS, forward consistency, and visualization reuse saved artifacts on CPU. Training resumes from compatible checkpoints; generation reuses compatible caches. Keep the completed checkpoint fixed throughout evaluation.

## Visualization

| Level | Required completed work | Example command |
| --- | --- | --- |
| One model | That model in one setting | `bash scripts/paper/m64n128/61_visualize_quantile_wasserstein.sh` |
| Within-setting comparison | Four models in that setting | `bash scripts/paper/m64n128/70_compare_models.sh` |
| Across-setting comparison | Four models in all three settings | `bash scripts/paper/90_export_results.sh` |

The first two levels are included in every setting's command list. The last is separate in [run_examples_comparisons.txt](run_examples_comparisons.txt).

All levels export **PNG/PDF metric plots** and **CSV summaries** for PR, ES/KS, forward consistency, and the flow-model NFE sweep. Numerical outputs include `all_metrics.csv`, `metrics_at_max_nfe.csv` (including F1), and `forward_consistency.csv`.

| Output location | Contents |
| --- | --- |
| `outputs/<setting>/visualizations/models/<variant>/` | One model's metric plots and reconstruction previews |
| `outputs/<setting>/visualizations/comparison/` | Within-setting model comparisons |
| `outputs/visualizations/all_settings/` | Combined metric plots and CSV summaries |

Model and within-setting folders contain `metrics/` and `reconstructions/`. Reconstruction plots show all saved draws and their mean, with ground-truth references and the original observation noise; case IDs are saved in `selected_cases.csv`. Input provenance is recorded alongside the outputs. Checkpoints, samples, and evaluation reports remain under `outputs/<setting>/`.

Optional method illustration: `bash scripts/paper/92_plot_transport_paths.sh` draws histogram/quantile interpolation paths without any trained model.

## Tests

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

These checks use small fixtures; they do not run full training or benchmarks.

## Paper and contact

*Quantile–Wasserstein Flow Matching for NMR Relaxation Spectrum Reconstruction*

Da Eun Lee and Byung-Woo Hong, Chung-Ang University. **Manuscript under review.**

For questions and bug reports, use [GitHub Issues](https://github.com/rechido/qwfm_nmr_reconstruction/issues).

## License

A license has not yet been specified for this repository.
