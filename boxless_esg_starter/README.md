# Boxless ESG — Optimized (Starter)

This is a **minimal, runnable scaffold** for your mask-first scene localization pipeline.
It runs **without heavy ML models** (uses simple fallbacks) so you can verify the end-to-end
plumbing and JSON output. Later, plug real models (MobileSAM, GroundingDINO, CLIP).

## Folder Map

boxless\_esg\_starter/
├─ src/boxless\_esg/
│ ├─ init.py
│ ├─ pipeline.py
│ ├─ utils.py
│ └─ proposals/
│ └─ selective\_search.py
├─ scripts/
│ └─ localize.py
├─ tests/
│ └─ test\_utils.py
└─ requirements.txt

\### Install

```bash

pip install -e .

\# For GroundingDINO:

pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git



