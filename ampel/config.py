"""Zentrale Konfiguration der Rheinschwimm-Ampel Basel."""

# Rhein bei Basel (Open-Meteo Flood API wählt die nächste Flusszelle)
BASEL = {"name": "basel", "lat": 47.56, "lon": 7.59}

# Wetterstationen im Einzugsgebiet des Rheins oberhalb Basel
CATCHMENT = [
    {"name": "basel", "lat": 47.56, "lon": 7.59},   # Basel
    {"name": "bern", "lat": 46.95, "lon": 7.45},    # Aare
    {"name": "luzern", "lat": 47.05, "lon": 8.31},  # Reuss
    {"name": "chur", "lat": 46.85, "lon": 9.53},    # Alpenrhein
]

TIMEZONE = "Europe/Zurich"

# Zeitraum
HISTORY_START = "2015-01-01"
TRAIN_END = "2024-12-31"  # Training bis hier, ab 2025 = Out-of-Sample Test

# Label-Definition: "Guter Rheinschwimm-Tag" (selbst gewählte Schwellen)
MIN_TMAX_C = 24.0          # Tageshöchsttemperatur in Basel
MAX_DISCHARGE_M3S = 1800.0  # Abfluss Rhein Basel (GloFAS Modell)

# Ampel-Schwellen auf der Modell-Wahrscheinlichkeit
GREEN_PROBA = 0.6
YELLOW_PROBA = 0.3

# Hopsworks
FEATURE_GROUP_NAME = "rhine_basel_daily"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "rheinschwimm_ampel_fv"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "rheinschwimm_ampel"
PREDICTION_FG_NAME = "rheinschwimm_predictions"
PREDICTION_FG_VERSION = 1

LABEL = "label_swim_tomorrow"

# Features, welche das Modell verwendet
BATCH_FEATURES = [
    "precip_catchment_3d_mm",     # aggregiert: 4 Orte x 3 Tage
    "discharge_m3s",              # Abfluss heute
    "discharge_trend_3d_m3s",     # aggregiert: Veränderung über 3 Tage
    "tmax_basel_3d_mean_c",       # aggregiert: Mittel Tmax über 3 Tage
    "day_of_year",
]
RT_FEATURES = [
    "temp_now_c",                 # RT: aktuelle Temperatur Basel
    "cloud_cover_now_pct",        # RT: aktuelle Bewölkung Basel
]
FEATURES = BATCH_FEATURES + RT_FEATURES

# Stunde (Lokalzeit), welche im Training als "jetzt" für die RT-Features gilt
RT_HOUR = 12
