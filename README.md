# BuildSync Gewerke-Benchmark

Cross-schedule construction schedule dashboard (Streamlit).

- **Gewerk cards** (`app_pages/gewerk_comparison.py`): one card per Gewerk, comparing
  every schedule that contains it - phase positions, active weeks by phase,
  delivered material volume, trucks per week against historical deliveries, and
  the work distributed over time. Reads `gewerk_data/` and `gewerk_core.py`,
  exported from the schedule-classification pipeline.
- **Schedule benchmark** (`schedule_benchmark/dashboard/app.py`): duration benchmark
  per gewerk, phase, task group and montage stage.

## Streamlit Cloud
- Main file: `streamlit_app.py`
- Branch: `main`
