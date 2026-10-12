"""BuildSync Gewerke-Benchmark - Streamlit Community Cloud entry point.

Two pages: the Gewerk cards (default) compare every schedule containing a
Gewerk - phases, durations, material volume and trucks over time; the schedule
benchmark is the cross-schedule duration dashboard in schedule_benchmark/.
"""

from __future__ import annotations

import streamlit as st

pg = st.navigation([
    st.Page("app_pages/gewerk_comparison.py", title="Gewerk cards", icon="🏗️", default=True),
    st.Page("schedule_benchmark/dashboard/app.py", title="Schedule benchmark", icon="📊"),
])
pg.run()
