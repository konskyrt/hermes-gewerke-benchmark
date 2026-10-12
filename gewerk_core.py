"""Pure Gewerk vocabularies and volume helpers - numpy/pandas only.

Kept free of the pipeline (no config, no schema imports) so the dashboard can
ship to Streamlit Cloud with just this file and the exported tables.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# Canonical phase order (earliest stage first). Used for sorting and as the
# fixed phase -> colour assignment in the dashboard.
PHASE_ORDER = ["Roh-/Grundarbeiten", "Grobmontage", "Ausführung",
               "Hauptausführung", "Feinmontage", "Endmontage",
               "Abschlussarbeiten"]

# Schedule label Gewerk -> factor-sheet Gewerk(e). Identity except where the
# schedules combine two factor trades under one label.
FINE_TO_FACTOR: Dict[str, List[str]] = {
    "Aufzüge/Fahrtreppen": ["Aufzüge", "Fahrtreppen"],
}

# Schedule label Gewerk -> factor_catalog.json Gewerk (the vocabulary of the
# historical data). Mirrors the judgement calls in discipline_map.json: scaffold
# and earthworks run on the shell programme, painters/floors/doors/lifts are
# Innenausbau, combined building services sit under Lüftungstechnik.
FINE_TO_CATALOG: Dict[str, str] = {
    "Rückbau/Schadstoffsanierung": "Rohbau",
    "Erdarbeiten/Baugrube": "Rohbau",
    "Werkleitungen/Tiefbau": "Rohbau",
    "Aussenanlagen": "Rohbau",
    "Rohbau/Baumeister": "Rohbau",
    "Betonbau": "Rohbau",
    "Mauerwerk": "Rohbau",
    "Stahlbau": "Rohbau",
    "Gerüst": "Rohbau",
    "Fassade": "Fassadenausbau",
    "Fenster": "Fassadenausbau",
    "Dach/Abdichtung": "Fassadenausbau",
    "Storen/Sonnenschutz": "Fassadenausbau",
    "Trockenbau/Gipser": "Trockenbau",
    "Decken": "Trockenbau",
    "Brandschutz": "Trockenbau",
    "Putz/Verputz": "Trockenbau",
    "Maler": "Innenausbau",
    "Bodenbeläge": "Innenausbau",
    "Unterlagsboden/Estrich": "Innenausbau",
    "Platten/Fliesen": "Innenausbau",
    "Schreiner/Innenausbau": "Innenausbau",
    "Türen/Tore": "Innenausbau",
    "Schliessanlage": "Innenausbau",
    "Glas/Trennwände": "Innenausbau",
    "Metallbau": "Innenausbau",
    "Aufzüge/Fahrtreppen": "Innenausbau",
    "Aufzüge": "Innenausbau",
    "Fahrtreppen": "Innenausbau",
    "Küchen": "Innenausbau",
    "Medizinaltechnik/Einrichtung": "Innenausbau",
    "Fertignasszellen": "Innenausbau",
    "Lüftung": "Lüftungstechnik",
    "Kälte": "Lüftungstechnik",
    "Rauch-/Wärmeabzug": "Lüftungstechnik",
    "TGA/HLKS": "Lüftungstechnik",
    "Heizung": "Heizung",
    "Sanitär": "Sanitär",
    "Rohrpost": "Sanitär",          # piped system; no better coarse home
    "Medizinalgase": "Gasanlagen",
    "Elektro": "Elektroinstallation",
    "Gebäudeautomation/MSRL": "Elektroinstallation",
    "Sprinkler": "Sprinkleranlage",
}

# Drivers that follow the building footprint (BGF / storeys) rather than the
# total floor area. Everything else scales linearly with BGF.
FOOTPRINT_DRIVERS = {"roof_m2", "site_m2", "excavation_m3"}
# Count-based drivers: linear scaling from the 1,000 m² reference is crude
# (2 lifts per 1,000 m²); the dashboard flags trades that depend on them.
UNIT_DRIVERS = {"door_units", "gate_units", "lift_units", "escalator_units",
                "kitchen_units", "medical_units", "wetroom_units"}

LKW_UTILIZATION = 0.75          # same constant as factor-based-analysis
NON_FREIGHT = {"Kein Fahrzeug", "PKW"}


# ---------------------------------------------------------------------------
# Volume estimates
# ---------------------------------------------------------------------------

def driver_quantities(drivers: dict, bgf_m2: float,
                      geschosse: Optional[float] = None) -> Dict[str, float]:
    """Scale the 1,000 m² reference driver quantities to a project.

    Area/volume drivers scale with BGF. Footprint drivers (roof, site,
    excavation) scale with BGF / storeys when the storey count is known; the
    reference building's implied storey count is floor_m2 / roof_m2.
    """
    ref_floor = drivers["floor_m2"]["example_quantity"]
    ref_storeys = ref_floor / drivers["roof_m2"]["example_quantity"]
    k = bgf_m2 / ref_floor
    out = {}
    for name, d in drivers.items():
        q = d["example_quantity"] * k
        if name in FOOTPRINT_DRIVERS and geschosse and geschosse > 0:
            q = d["example_quantity"] * k * ref_storeys / geschosse
        out[name] = q
    return out


def estimate_volumes(comp: pd.DataFrame, drivers: dict, gewerk_label: str,
                     bgf_m2: float, geschosse: Optional[float] = None) -> pd.DataFrame:
    """Delivered m³ per component for one schedule Gewerk and project size."""
    names = FINE_TO_FACTOR.get(gewerk_label, [gewerk_label])
    sub = comp[comp["gewerk"].isin(names)].copy()
    if sub.empty:
        return sub
    q = driver_quantities(drivers, bgf_m2, geschosse)
    sub["driver_qty"] = sub["driver"].map(q)
    sub["medium_m3"] = sub["medium_delivered_m3_per_unit"] * sub["driver_qty"]
    sub["high_m3"] = sub["high_delivered_m3_per_unit"] * sub["driver_qty"]
    return sub
