# BuildSync Gewerke-Benchmark

Cross-schedule construction schedule dashboard (Streamlit): one card per Gewerk,
comparing every schedule that contains it - phases by project (duration, timing,
tasks) and the Gewerk's flow over time (m³, LKW and m² BGF per week, cumulative
progress). Reads `gewerk_data/` and `gewerk_core.py`, exported from the
schedule-classification pipeline.

## Streamlit Cloud
- Main file: `streamlit_app.py`
- Branch: `main`
