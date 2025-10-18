@echo off
REM Example: forecast from checkpoint and context CSV
python -m fx_timesfm.infer --ckpt checkpoints\model_best.pt --horizon 256 --context_csv context.csv --device cpu
