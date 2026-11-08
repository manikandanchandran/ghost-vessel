# Ghost Vessels — corrected, reproducible source package (not yet final evidence)

**Manuscript:** OJ-ITS-2026-01-0041, *Ghost Vessels: AI-Generated Fake AIS Signals and the Future of Maritime Cyber Threats*.

## Inputs needed (not supplied with the ZIP)

1. Use the public, licensed TrAISformer (Nguyen & Fablet, *IEEE Access*, 2024) preprocessed Danish Maritime Authority (DMA) data: <https://github.com/CIA-Oceanix/TrAISformer>, the cargo/tanker subset for 55.5–58.0°N / 10.3–13.0°E.
2. Put the original `ct_dma_train.pkl`, `ct_dma_valid.pkl`, `ct_dma_test.pkl` and `dma_coastline_polygons.pkl` files in **`raw/`**. Confirm upstream licenses/redistribution terms.
3. From the package root, run `python prep.py` to generate **`data.npz`** and **`coast.pkl`**. These are excluded because the original archive did not include the inputs.

## Environment

Use Python 3.11–3.13, NumPy 2.x, Shapely >=2.0, PyTorch 2.x, scikit-learn >=1.5, SciPy, and Matplotlib. Version ranges and install instructions are in `requirements.txt`. Environment versions can influence nondeterministic training and therefore the exact samples. Tamarin 1.10 + Maude 3.5 are required only for rechecking symbolic proofs.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python prep.py
```

## Pipeline — run each model separately

Run these commands from the extracted ZIP folder (not the `raw/` folder):

```bash
python gen.py baselines
python gen.py lstm
python gen.py transformer
python gen.py vae
# Only if you want to retrain; provided checkpoints come from the original 4,000-step runs:
# python gen.py gan 4000
# python gen.py kingan 4000
python kin_noise.py
python kin_land.py
python evaluate.py realism
python evaluate.py detect
python evaluate.py ablate
python evaluate.py cross
python oppoint.py
python protocol/budget.py
python mkresults.py
python mkfigs.py traj game
python verify_submission.py
```

**Important:** `kin_land.py` uses the existing `ckpt_kingan.pt` (included) and calibrations in `results/train_kinnoise.json` (included). It needs `data.npz`, `coast.pkl`, and then rewrites active `samples/kinland.npz` and `results/train_kinland.json`. After that, rerun **all affected result files** listed above, including any experiment whose generator samples were regenerated. Use fixed seeds and confirm quantitative reproducibility; never edit generated JSON manually.

`oppoint.py` uses validation-selected thresholds for the statistical detectors. Coastline detection is a **fixed yes/no spatial rule** (any segment or position intersects land). Its validation and realized test FPRs are reported; it cannot be tuned to exactly 1% without changing the rule. The combined detector is the logical OR of that fixed coastline rule and the validation-calibrated physics detector. Check that the manuscript describes this distinction accurately.

