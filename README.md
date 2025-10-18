# ZFXT : Minimal Decoder-Only Foundation Model for FX Forecasting

This repo provides a from-scratch, clean PyTorch implementation of a TimesFM-inspired, decoder-only transformer for time-series forecasting with patch tokens—specialized for foreign exchange (FX) rates.

Key ideas:
- Tokens are contiguous non-overlapping patches of the time series (input_patch_len p=32).
- The model predicts long future patches per token (output_patch_len h=128).
- Decoder-only causal attention over token sequence; Residual MLP blocks before/after the transformer.
- Reversible first-patch normalization for stable training and easy de-normalization.
- Variable-context masking by sampling r∈[0,p-1] points masked in the first input patch.

Configs provide small (~17M), medium (~70M), and large (~200M) presets.

## Install

```powershell
pip install -r requirements.txt
```

## Quick start (Norges Bank FX data)

1) Fetch and prepare FX data (1970–1999 daily rates across multiple currencies vs NOK) using the official API:

```powershell
python scripts/fetch_norges_bank_exr.py --outdir data --start 1970-01-01 --end 1999-12-31 --locale no
```

This writes:
- data/norges_bank_exr_long.csv (raw long format)
- data/norges_bank_exr_wide.csv (optional, human readable)
- data/train.csv, data/val.csv, data/test.csv in the expected schema:
  - timestamp, symbol, price

2) Train the small (~17M) model on daily data:

```powershell
python -m fx_timesfm.train --config fx_timesfm/config/small_nb_exr.yaml
```

3) Forecast from a checkpoint given a context CSV (one numeric column for values):

```powershell
python -m fx_timesfm.infer --ckpt runs\nb_1970_1999\best.ckpt --horizon 256 --context_csv path\to\context.csv --device cpu
```

4) Evaluate last-window performance on the validation split:

```powershell
python -m fx_timesfm.bench_fx --config fx_timesfm/config/small_nb_exr.yaml --split val
```

## Configuration

See `fx_timesfm/config/` for examples. Important fields:
- data:
  - train_paths/val_paths/test_paths: list of CSV/Parquet files.
  - timestamp_col/symbol_col/price_col
  - freq, resample, fill_method
  - context_len (≤512), input_patch_len p=32, output_patch_len h=128, horizon
  - batch_size, num_workers
- model:
  - preset: small|medium|large, or override d_model/n_layers/n_heads
  - dropout, quantiles, quantile_levels
- trainer:
  - epochs, lr, weight_decay, betas, grad_clip_norm
  - cosine_decay, warmup_steps
  - early_stop_patience, ckpt_dir

## Data pipeline

- Load CSV/Parquet, filter by symbols (optional), parse timestamps, and sort.
- Resample per symbol to uniform cadence (e.g., 1D) with forward-fill/backfill/interpolate.
- Split into rolling windows of length L (≤512) per item.
- Patchify: break windows into non-overlapping p=32 tokens; pad last patch as needed; keep per-point pad mask.
- Collate batches with token padding to the longest T among batch items; produce attention padding masks.

## Masking strategy (training)

- For each sample, draw r ∈ [0, p-1]. Mask the first r points in the first input patch.
- Combined with padding masks, this exposes every possible effective context length across the dataset.

## Model

- Input residual block: MLP over patch values → token embeddings of size d_model.
- Positional encodings: sinusoidal, added to token embeddings.
- Decoder-only transformer stack: pre-norm LayerNorm → Multi-Head causal Self-Attention → FFN (d_model→4*d_model→d_model) with dropout.
- Output head: map each token to an h=128-length future patch. Optional quantile head for τ∈{0.1,0.5,0.9}.

## Training

- Loss: default point MSE in normalized space. Optional quantile pinball combined loss under a flag.
- Optimizer: AdamW with cosine LR decay + warmup, gradient clipping.
- Validation: MAE on de-normalized predictions. Early stopping monitors the monthly MAE gap vs a driftless RW baseline (fallback to negative validation MAE if macro eval is undefined).
- Checkpoints include a config hash and a ModelCard for reproducibility.

## Inference

- Autoregressive in h-sized blocks: after each step, append a token formed from the first p points of predicted patch.
- Anchoring: add `--anchor_to_last true` to make the first forecast equal the last observed value (removes initial level gap).
- CLI returns a CSV with the forecast and a JSON with metadata.

### Walk-forward and quarterly evaluation (Norges Bank)

- Walk-forward (e.g., monthly end):

```powershell
python -m fx_timesfm.scripts.walk_forward_nb --ckpt runs\nb_1970_1999\best.ckpt ^
  --base EUR --quote NOK --start 2000-01-01 --end 2024-12-31 --freq ME --context_len 512 ^
  --out runs\walk_forward_eurnok_2000_2024
```

- Quarterly cutoff evaluation (origin = day before quarter start; target = quarter end):

```powershell
python -m fx_timesfm.scripts.quarterly_eval_nb --ckpt runs\nb_1970_1999\best.ckpt ^
  --base EUR --quote NOK --start 2000-01-01 --end 2024-12-31 --context_len 512 ^
  --out runs\quarterly_eval_eurnok_2000_2024
```

Both scripts save CSV + plot under `runs/...` and compare the model to a driftless RW baseline.

## Metrics

- Implemented: MAE, sMAPE, msMAPE, MASE (per-series seasonal naive scaling), and Diebold–Mariano (two-sided, abs or squared loss) with Newey–West variance.

### Quarterly metrics and DM test on a CSV

Given a quarterly CSV with columns `model`, `rw`, `actual` (like those produced by the scripts), compute summary metrics and DM p-value:

```powershell
python -m fx_timesfm.scripts.metrics_quarterly --csv "path\to\quarterly_eval.csv"
```

Notes:
- DM uses two-sided test with h=1 and Newey–West lag default h−1; you can override with `--nw_lag 0`.
- Very small p-values are printed in scientific notation.

### Covariate-adjusted walk-forward

Run a leakage-safe, origin-mode, covariate residual adjustment over daily walk-forward and aggregate to quarter-end:

```powershell
python -m fx_timesfm.scripts.walk_forward_covariates ^
  --ckpt runs\nb_1970_1999\best.ckpt ^
  --csv dataset_endogene.csv ^
  --context_len 512 ^
  --covariates Q,d_pi,dI_t ^
  --out runs\wf_endogene ^
  --anchor_to_last true ^
  --to_quarter_end ^
  --cov_lag 1 ^
  --cov_strategy carryforward
```

## Model sizes

- small (~17M): d_model=512, n_layers=8, n_heads=8
- medium (~70M approx): configurable via config (example preset included)
- large (~200M target): d_model=1280, n_layers=20, n_heads=16

VRAM requirements depend on sequence length, batch size, and precision. For first experiments, try small with BS=8–16 on a single GPU or CPU.

## Tests

Run unit tests (synthetic fixtures):

```powershell
pytest -q
```

Tests cover: patching, masking, forward shapes, and normalization round-trip.

## Swapping datasets

- Replace `data/*.csv` paths in configs with your own CSV/Parquet files.
- Ensure schema: timestamp (timezone-aware or parseable), symbol, price.
- Adjust `freq`, `resample`, and `fill_method` for your cadence.

## Troubleshooting

- If imports fail in tests, ensure project root is on PYTHONPATH (we add it via `conftest.py`).
- For CPU-only environments, set device to `cpu` in config or CLI `--device cpu`.
- For Windows PowerShell, example scripts are `.sh` but contain `python` one-liners compatible with `cmd`/PowerShell; invoke them directly or copy the command.

## License

MIT — see `LICENSE`. The code uses only official, publicly available APIs and does not include proprietary data.

## Publishing and repo hygiene

This repo is configured to avoid committing large/local artifacts:
- `.gitignore` excludes environments (`.venv/`), caches, `runs/`, `data/`, `checkpoints/`, and large file types (`*.csv`, `*.parquet`).
- `Resultater/**` is ignored except for images (`.png/.jpg/.jpeg/.svg`) so you can include selected figures.
- See `PUBLISHING.md` for a quick pre-publish checklist.

## Repository layout

```
fx_timesfm/           # Library code (models, data utils, evaluation)
scripts/              # CLI scripts for training/evaluation
docs/                 # Project docs (LaTeX, theory, results writeups)
examples/             # Small runnable examples / scratch
Resultater/           # Optional figures for papers (images only committed)
README.md             # This file
LICENSE               # MIT license
requirements.txt      # Python dependencies
```

## Example: EUR/NOK inference demo (Norges Bank data)

One-line demo that fetches, forecasts, and plots:

```powershell
python -m fx_timesfm.infer ^
  --ckpt runs\nb_1970_1999\best.ckpt ^
  --demo_norgesbank true ^
  --horizon 128 ^
  --anchor_to_last true
```

Output:
- Downloads EUR/NOK spot rates
- Generates a 128‑day forecast
- Saves plot to `runs/plots/eurnok_demo.png`

---

See also: `documentation.txt` for a math-formalized description of the full pipeline (architecture, training, inference, evaluation).

