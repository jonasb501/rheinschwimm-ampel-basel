"""3. Inference Pipeline: RT-Daten + neueste Features aus der Feature View -> Modell -> Ampel.

Beispiele:
    uv run python inference_pipeline.py                    # neueste Features + RT-Features live
    uv run python inference_pipeline.py --date 2025-06-14  # Backtest für einen beliebigen Tag
    uv run python inference_pipeline.py --no-log           # Prediction nicht in den Feature Store schreiben
"""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import hopsworks
import joblib
import pandas as pd
from dotenv import load_dotenv

from ampel import config as c
from ampel.openmeteo import fetch_rt_history, fetch_rt_live


def ampel(proba: float) -> str:
    if proba >= c.GREEN_PROBA:
        return "🟢 GRÜN  – ab in den Rhein!"
    if proba >= c.YELLOW_PROBA:
        return "🟡 GELB  – unsicher, Wetter & Pegel nochmals checken"
    return "🔴 ROT   – lieber nicht (zu kalt oder zu viel Wasser)"


def load_batch_features(fv, day: date | None) -> pd.DataFrame:
    """Batch-Features aus der Feature View laden.

    day=None: neuester verfügbarer Tag (max. 7 Tage zurück).
    Das Zeitfenster ist bewusst grosszügig gewählt (Zeitzonen), danach wird exakt auf den Tag gefiltert.
    """
    target = day or date.today()
    batch = fv.get_batch_data(
        start_time=datetime.combine(target - timedelta(days=7 if day is None else 1), datetime.min.time()),
        end_time=datetime.combine(target + timedelta(days=1), datetime.max.time()),
        primary_key=True,
    )
    if batch.empty:
        return batch
    # Hopsworks liefert den Primary Key mit Präfix (<projekt>_<fg>_<version>_date_id)
    pk_col = next(col for col in batch.columns if col.endswith("date_id"))
    batch["day"] = pd.to_datetime(batch[pk_col]).dt.date
    batch = batch[batch["day"] == day] if day else batch[batch["day"] == batch["day"].max()]
    return batch.tail(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Backtest-Tag (YYYY-MM-DD). Ohne Angabe: neueste Features + Live-RT-Daten")
    parser.add_argument("--no-log", action="store_true", help="Prediction nicht in die Feature Group schreiben")
    args = parser.parse_args()
    backtest_day = date.fromisoformat(args.date) if args.date else None

    load_dotenv()
    project = hopsworks.login()
    fs = project.get_feature_store()

    # 1. Feature View holen und die neuesten Batch-Features (bzw. die des Backtest-Tages) laden
    fv = fs.get_feature_view(c.FEATURE_VIEW_NAME, version=c.FEATURE_VIEW_VERSION)
    batch = load_batch_features(fv, backtest_day)
    if batch.empty:
        raise SystemExit("Keine passenden Features im Feature Store. Zuerst feature_pipeline.py laufen lassen.")
    day = batch["day"].iloc[0]
    live = backtest_day is None
    if live and day != date.today():
        print(f"⚠️  Neueste Features sind vom {day} (nicht von heute). Tipp: feature_pipeline.py --start <Datum>")

    # 2. RT-Features: live (Forecast API) oder für Backtests historisch (Archive API)
    if live:
        rt = fetch_rt_live()
        rt_source = f"live {rt.pop('time')}"
    else:
        rt = fetch_rt_history(day.isoformat(), day.isoformat()).iloc[0][c.RT_FEATURES].to_dict()
        rt_source = f"{day} {c.RT_HOUR}:00"
    X = batch[c.FEATURES].copy()
    for k, v in rt.items():
        X[k] = v  # RT-Features ersetzen die gespeicherten Werte

    # 3. Modell aus der Model Registry herunterladen
    mr = project.get_model_registry()
    hs_model = mr.get_best_model(c.MODEL_NAME, "roc_auc", "max")
    model = joblib.load(Path(hs_model.download()) / "model.pkl")

    # 4. Prediction
    proba = float(model.predict_proba(X[c.FEATURES])[:, 1][0])
    light = ampel(proba)

    print("\n" + "=" * 62)
    print(f"  RHEINSCHWIMM-AMPEL BASEL  –  Prognose für {day + timedelta(days=1)}")
    print(f"  (Modell v{hs_model.version}, Features vom {day}, RT {rt_source})")
    print("=" * 62)
    print(f"  Abfluss Rhein heute:           {X['discharge_m3s'].iloc[0]:8.0f} m³/s")
    print(f"  Abfluss-Trend 3 Tage:          {X['discharge_trend_3d_m3s'].iloc[0]:+8.0f} m³/s")
    print(f"  Regen Einzugsgebiet 3 Tage:    {X['precip_catchment_3d_mm'].iloc[0]:8.1f} mm")
    print(f"  Tmax Basel Ø 3 Tage:           {X['tmax_basel_3d_mean_c'].iloc[0]:8.1f} °C")
    print(f"  Temperatur jetzt (RT):         {X['temp_now_c'].iloc[0]:8.1f} °C")
    print(f"  Bewölkung jetzt (RT):          {X['cloud_cover_now_pct'].iloc[0]:8.0f} %")
    print("-" * 62)
    print(f"  Wahrscheinlichkeit Schwimmtag: {proba:8.0%}")
    print(f"  {light}")

    # Backtest: tatsächliches Ergebnis aus dem Feature Store zum Vergleich
    if not live:
        fg = fs.get_feature_group(c.FEATURE_GROUP_NAME, version=c.FEATURE_GROUP_VERSION)
        truth = fg.filter(fg.date_id == day.isoformat()).read()
        if not truth.empty and pd.notna(truth[c.LABEL].iloc[0]):
            actual = "Schwimmtag ✅" if int(truth[c.LABEL].iloc[0]) == 1 else "kein Schwimmtag ❌"
            print(f"  Tatsächlich:                   {actual}")
    print("=" * 62)

    # 5. Extra: Prediction inkl. verwendeter RT-Features zurück in den Feature Store schreiben
    #    (Basis für Monitoring und späteren Vergleich mit der Ground Truth)
    if not args.no_log:
        log = X[c.FEATURES].copy()
        log.insert(0, "date_id", day.isoformat())
        log["prediction_time"] = pd.Timestamp.now().floor("s")
        log["is_backtest"] = not live
        log["model_version"] = int(hs_model.version)
        log["proba_swim"] = proba
        log["ampel"] = light.split()[1]
        pred_fg = fs.get_or_create_feature_group(
            name=c.PREDICTION_FG_NAME,
            version=c.PREDICTION_FG_VERSION,
            primary_key=["date_id", "is_backtest"],
            event_time="prediction_time",
            description="Log der Rheinschwimm-Ampel: Features, RT-Werte und Prediction pro Lauf",
            online_enabled=False,
        )
        pred_fg.insert(log, write_options={"wait_for_job": False})
        print(f"Prediction in Feature Group '{c.PREDICTION_FG_NAME}' geloggt.")


if __name__ == "__main__":
    main()
