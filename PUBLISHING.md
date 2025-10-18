# Publishing checklist

Use this checklist to prepare the repository for GitHub.

- Ensure a clean `.gitignore` excludes large, local, or sensitive files:
  - Environments: `.venv/`, `env/`
  - Caches/logs/outputs: `__pycache__/`, `runs/`, `outputs/`, `logs/`
  - Data/checkpoints: `data/`, `checkpoints/`, `data_nb_70_99/`
  - Heavy files: `*.csv`, `*.parquet`
  - Results: `Resultater/**` except figures (`.png/.jpg/.jpeg/.svg`)
- Include only publishable artifacts:
  - Source: `fx_timesfm/`, `scripts/`, `README.md`, `requirements.txt`
  - Optional docs: `revised.tex`, `results.tex`, `fx_architecture.mmd`
  - Figures (optional): selected images under `Resultater/`
- Add a LICENSE (MIT suggested) — see template below.
- Verify no secrets (API keys, tokens) in code or configs.
- Test a fresh install:
  1. `python -m venv .venv` then activate
  2. `pip install -r requirements.txt`
  3. Run a small script (e.g., tests or a dry run of evaluation)
- Create a clean history: commit with clear messages; squash if needed.

## MIT License (template)

Create `LICENSE` with the following:

```
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
