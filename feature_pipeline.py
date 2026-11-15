"""1. Feature Pipeline: Open-Meteo -> Features + Label -> Hopsworks Feature Group.

Beispiel:
    uv run python feature_pipeline.py                 # Backfill 2015 bis heute
    uv run python feature_pipeline.py --start 2026-10-01   # nur neuere Tage (Daily Run)
"""

import argparse
from datetime import date, timedelta

import hopsworks
from dotenv import load_dotenv

from ampel import config as c
from ampel.features import build_features


def validate(df) -> None:
    """Einfache Datenvalidierung vor dem Insert (leichtgewichtige Alternative zu Great Expectations)."""
    errors = []
    if df.empty:
        errors.append("keine Daten")
    if df["date_id"].duplicated().any():
        errors.append("doppelte date_id (Primary Key)")
    if df[c.FEATURES].isna().any().any():
        errors.append(f"fehlende Werte in Features: {df[c.FEATURES].isna().sum()[lambda s: s > 0].to_dict()}")
    if not df["discharge_m3s"].between(50, 6000).all():
        errors.append("Abfluss ausserhalb 50-6000 m³/s")
    if not df["temp_now_c"].between(-30, 45).all():
        errors.append("Temperatur ausserhalb -30..45 °C")
    if not df["cloud_cover_now_pct"].between(0, 100).all():
        errors.append("Bewölkung ausserhalb 0..100 %")
    if df[c.LABEL].isna().sum() > 1:
        errors.append("mehr als ein Tag ohne Label (nur der neueste Tag darf ohne Label sein)")
    if errors:
        raise SystemExit("Datenvalidierung fehlgeschlagen:\n- " + "\n- ".join(errors))
    print("Datenvalidierung OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=c.HISTORY_START)
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()

    # 3 Tage Vorlauf, damit die Rolling Windows ab --start vollständig sind
    fetch_start = (date.fromisoformat(args.start) - timedelta(days=3)).isoformat()
    df = build_features(fetch_start, args.end)
    df = df[df["date"] >= args.start]
    validate(df)
    print(f"{len(df)} Tage mit Features von {df['date_id'].min()} bis {df['date_id'].max()}")
    print(f"Label-Verteilung: {df[c.LABEL].value_counts(dropna=False).to_dict()}")
    print(df.tail(3).to_string())

    load_dotenv()
    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=c.FEATURE_GROUP_NAME,
        version=c.FEATURE_GROUP_VERSION,
        primary_key=["date_id"],
        event_time="date",
        description="Rhein Basel: Abfluss, Niederschlag im Einzugsgebiet, Wetter Basel + Label Rheinschwimm-Tag morgen",
        online_enabled=False,
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Feature Group '{c.FEATURE_GROUP_NAME}' v{c.FEATURE_GROUP_VERSION} aktualisiert.")


if __name__ == "__main__":
    main()
