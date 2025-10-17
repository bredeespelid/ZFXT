# foundation-timesfm-decoder

A production-grade, academically reproducible decoder-only transformer for time-series forecasting inspired by TimesFM (Google Research, 2024).

- Causal autoregressive GPT-like architecture (decoder-only)
- Pretrained on Norges Bank FX dataset (1980–1999), 6961×206
- Self-supervised masked-patch reconstruction objective
- Zero-shot forecasting on unseen datasets (univariate or multivariate)

## Architecture Overview

- Input projection: Linear(206 → 256)
- 6 decoder blocks, 8 heads, d_model=256, ffn_dim=1024, dropout=0.1
- Learned positional embeddings with context_len=512
- Output projection: Linear(256 → 206)
- Loss: Masked MSE over randomly masked temporal patches (TimesFM-style pretraining)

## Dataset

- Format: `train_df_timeseries.parquet` — wide numeric DataFrame with DatetimeIndex
- Period: 1980-12-10 → 1999-12-31 (≈ 6961 daily)
- Features: 206 columns (e.g., FX rates, returns, vol/momo, Fourier, PCA, calendar)
- No missing values, daily frequency

## Pretraining

- Objective: Mask 20–30% of temporal patches (patch_len=32, stride=8), reconstruct masked values with MSE
- Optimizer: AdamW (lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
- Scheduler: Cosine annealing with warmup
- Mixed precision: fp16 where available
- Early stopping on validation MSE
- Checkpoint and normalization stats saved under `checkpoints/`

## Zero-Shot Forecasting

- Accepts .parquet or .csv with a datetime column and one or more numeric columns
- Automatically infers datetime and frequency
- Uses pretrained μ/σ for normalization
- Handles univariate and multivariate
- Forecast horizons: 30, 90, 180 via `--H`
- Outputs: `outputs/forecast_YYYYMMDD.csv` with [timestamp, prediction, lower_ci, upper_ci]
- Saves latent embeddings for interpretability

## Install

```
make setup
```

(Windows PowerShell users: `make` may not be available; you can run the commands manually.)

## Train

Place `train_df_timeseries.parquet` at repository root, then:

```
make train
```

This logs metrics to TensorBoard under `runs/`.

## Forecast

```
make forecast DATA_PATH=user_dataset.parquet H=90
```

Or run manually:

```
python -m src.infer.forecast --DATA_PATH user_dataset.parquet --H 90
```

## Evaluation & Metrics

Use your ground truth to compute MAE, RMSE, MAPE, CRPS (notebook demo provided). Visualizations include predictions vs truth with confidence bands.

## Project Structure

See the repository tree for code layout under `src/`. Configs in `configs/default.yaml`.

## License

MIT
