# Road Accident Analysis — Starter Kit

This repo scaffolds a complete **EDA → ML → Dashboard** workflow for a Road Accident Analysis project.

## Folder Structure
```
road_accident_analysis_starter/
├─ data/
│  ├─ raw/          # put original CSV(s) here
│  ├─ processed/    # cleaned, feature-engineered CSV(s)
│  └─ external/     # keep geojson/shapefiles, lookup tables (holidays, codes)
├─ models/          # trained model, encoders, metrics
├─ notebooks/       # your exploratory notebooks
├─ reports/
│  └─ figures/      # exported plots for your report
├─ src/             # reproducible scripts
└─ app.py           # Streamlit dashboard
```

## Quickstart
1) **Install packages** (preferably in a fresh venv):
```bash
pip install -r requirements.txt
```
2) **Place your dataset** in `data/raw/` as `accidents_raw.csv` (or update the paths in scripts).
3) **Clean & feature engineer**:
```bash
python src/preprocess.py
```
4) **Train a model** (predict accident severity):
```bash
python src/train_model.py
```
5) **Run the dashboard**:
```bash
streamlit run app.py
```

## Data Notes
- Expected minimal columns (rename yours to match or edit code):  
  - `date` (YYYY-MM-DD), `time` (HH:MM or HH:MM:SS), `state`, `city`, `vehicle_type`, `weather`, `road_type`, `severity` (Low/Medium/High).
- Optional helpful columns: `lat`, `lon`, `cause`, `speed_limit`, `light_conditions`.
- If you lack coordinates, you can still do **state-level choropleths** by joining to a states GeoJSON.

## Geospatial
- Put `india_states.geojson` (or shapefile converted to GeoJSON) in `data/external/`. Update `GEOJSON_PATH` in `app.py` if you use a different file.

## Report Exports
- Save final figures to `reports/figures/` from notebooks or scripts so you can paste them into your internship report.

## Tips
- Start simple: EDA → Baseline model → Iterate.
- Log decisions (data cleaning, encoding choices) in your report.
- Keep `random_state=42` for reproducibility.
