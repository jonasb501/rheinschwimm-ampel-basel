"""Feature Engineering: aus Rohdaten wird ein Dataframe mit Features + Label."""

import pandas as pd

from ampel.config import CATCHMENT, LABEL, MAX_DISCHARGE_M3S, MIN_TMAX_C
from ampel.openmeteo import fetch_catchment_weather, fetch_discharge, fetch_rt_history


def build_features(start: str, end: str) -> pd.DataFrame:
    weather = fetch_catchment_weather(start, end)
    discharge = fetch_discharge(start, end)
    rt = fetch_rt_history(start, end)

    df = weather.merge(discharge, on="date").merge(rt, on="date").sort_values("date")

    # Aggregiertes Feature über mehrere Datenquellen UND mehrere Zeitschritte:
    # mittlerer Niederschlag im Einzugsgebiet, aufsummiert über die letzten 3 Tage
    precip_cols = [f"precip_{c['name']}_mm" for c in CATCHMENT]
    df["precip_catchment_mm"] = df[precip_cols].mean(axis=1)
    df["precip_catchment_3d_mm"] = df["precip_catchment_mm"].rolling(3).sum()

    # Aggregierte Features über mehrere Zeitschritte
    df["discharge_trend_3d_m3s"] = df["discharge_m3s"].diff(3)
    df["tmax_basel_3d_mean_c"] = df["tmax_basel_c"].rolling(3).mean()
    df["day_of_year"] = df["date"].dt.dayofyear

    # Label: Ist MORGEN ein guter Rheinschwimm-Tag?
    tmax_tomorrow = df["tmax_basel_c"].shift(-1)
    discharge_tomorrow = df["discharge_m3s"].shift(-1)
    df[LABEL] = ((tmax_tomorrow >= MIN_TMAX_C) & (discharge_tomorrow <= MAX_DISCHARGE_M3S)).astype("Int64")
    df.loc[tmax_tomorrow.isna() | discharge_tomorrow.isna(), LABEL] = pd.NA  # Zukunft noch unbekannt

    df = df.dropna(subset=["precip_catchment_3d_mm", "discharge_trend_3d_m3s", "tmax_basel_3d_mean_c"])
    df["date_id"] = df["date"].dt.strftime("%Y-%m-%d")  # Primary Key

    cols = ["date_id", "date", "precip_catchment_3d_mm", "discharge_m3s", "discharge_trend_3d_m3s",
            "tmax_basel_3d_mean_c", "day_of_year", "temp_now_c", "cloud_cover_now_pct", LABEL]
    return df[cols].reset_index(drop=True)
