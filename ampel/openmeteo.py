"""Abruf der Rohdaten von Open-Meteo (kein API-Key nötig)."""

import time

import pandas as pd
import requests

from ampel.config import BASEL, CATCHMENT, RT_HOUR, TIMEZONE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"


def _get(url: str, params: dict, retries: int = 3) -> dict | list:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries:
                raise
            print(f"Open-Meteo Fehler ({e.__class__.__name__}), neuer Versuch {attempt + 1}/{retries} ...")
            time.sleep(5 * attempt)


def fetch_discharge(start: str, end: str) -> pd.DataFrame:
    """Täglicher Abfluss des Rheins bei Basel (GloFAS)."""
    data = _get(FLOOD_URL, {
        "latitude": BASEL["lat"], "longitude": BASEL["lon"],
        "daily": "river_discharge", "start_date": start, "end_date": end,
    })
    return pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "discharge_m3s": data["daily"]["river_discharge"],
    })


def fetch_catchment_weather(start: str, end: str) -> pd.DataFrame:
    """Täglicher Niederschlag an allen Orten im Einzugsgebiet + Tmax Basel."""
    data = _get(ARCHIVE_URL, {
        "latitude": ",".join(str(c["lat"]) for c in CATCHMENT),
        "longitude": ",".join(str(c["lon"]) for c in CATCHMENT),
        "daily": "precipitation_sum,temperature_2m_max",
        "start_date": start, "end_date": end, "timezone": TIMEZONE,
    })
    df = pd.DataFrame({"date": pd.to_datetime(data[0]["daily"]["time"])})
    for loc, d in zip(CATCHMENT, data):
        df[f"precip_{loc['name']}_mm"] = d["daily"]["precipitation_sum"]
    df["tmax_basel_c"] = data[0]["daily"]["temperature_2m_max"]
    return df


def fetch_rt_history(start: str, end: str) -> pd.DataFrame:
    """Historische 'Echtzeit'-Werte in Basel: Temperatur und Bewölkung um RT_HOUR Uhr."""
    data = _get(ARCHIVE_URL, {
        "latitude": BASEL["lat"], "longitude": BASEL["lon"],
        "hourly": "temperature_2m,cloud_cover",
        "start_date": start, "end_date": end, "timezone": TIMEZONE,
    })
    h = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temp_now_c": data["hourly"]["temperature_2m"],
        "cloud_cover_now_pct": data["hourly"]["cloud_cover"],
    })
    h = h[h["time"].dt.hour == RT_HOUR].copy()
    h["date"] = h["time"].dt.normalize()
    return h.drop(columns="time")


def fetch_rt_live() -> dict:
    """Aktuelle Werte in Basel (Forecast API, 'current')."""
    data = _get(FORECAST_URL, {
        "latitude": BASEL["lat"], "longitude": BASEL["lon"],
        "current": "temperature_2m,cloud_cover", "timezone": TIMEZONE,
    })
    return {
        "temp_now_c": data["current"]["temperature_2m"],
        "cloud_cover_now_pct": data["current"]["cloud_cover"],
        "time": data["current"]["time"],
    }
