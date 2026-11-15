"""2. Training Pipeline: Feature Group -> Feature View -> Training Data -> Modell -> Model Registry.

Zeitbasierter Split: Training 2015-2024, Test ab 2025 (Out-of-Sample).

Beispiel:
    uv run python training_pipeline.py
"""

import json
from datetime import date, timedelta
from pathlib import Path

import hopsworks
import joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from ampel import config as c

MODEL_DIR = Path("model")


def main() -> None:
    load_dotenv()
    project = hopsworks.login()
    fs = project.get_feature_store()

    # 1. Feature Group laden
    fg = fs.get_feature_group(c.FEATURE_GROUP_NAME, version=c.FEATURE_GROUP_VERSION)

    # 2. Feature View: Features + Label auswählen
    fv = fs.get_or_create_feature_view(
        name=c.FEATURE_VIEW_NAME,
        version=c.FEATURE_VIEW_VERSION,
        query=fg.select(c.FEATURES + [c.LABEL]),
        labels=[c.LABEL],
        description="Features für die Rheinschwimm-Ampel Basel",
    )

    # 3. Training Dataset mit zeitbasiertem Split (über die Event Time 'date')
    X_train, X_test, y_train, y_test = fv.train_test_split(
        train_start=c.HISTORY_START, train_end=c.TRAIN_END,
        test_start="2025-01-01", test_end=(date.today() + timedelta(days=1)).isoformat(),
        description="Train 2015-2024, Test ab 2025",
    )
    # Neuester Tag hat noch kein Label (morgen unbekannt) -> entfernen
    train_mask, test_mask = y_train[c.LABEL].notna(), y_test[c.LABEL].notna()
    X_train, y_train = X_train[train_mask][c.FEATURES], y_train[train_mask][c.LABEL].astype(int)
    X_test, y_test = X_test[test_mask][c.FEATURES], y_test[test_mask][c.LABEL].astype(int)
    print(f"Train: {len(X_train)} Tage, Test: {len(X_test)} Tage")

    # 4. Modell trainieren und evaluieren
    model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
    }
    print(classification_report(y_test, pred, target_names=["kein Schwimmtag", "Schwimmtag"]))
    print("Confusion Matrix (Zeilen = Realität, Spalten = Prediction):")
    print(confusion_matrix(y_test, pred))
    print("Metriken:", metrics)

    importance = dict(sorted(zip(c.FEATURES, model.feature_importances_.round(3)), key=lambda kv: -kv[1]))
    print("Feature Importance:", importance)

    # 5. Lokal speichern und in die Hopsworks Model Registry hochladen
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.pkl")
    (MODEL_DIR / "features.json").write_text(json.dumps(c.FEATURES))
    (MODEL_DIR / "feature_importance.json").write_text(json.dumps({k: float(v) for k, v in importance.items()}, indent=2))
    (MODEL_DIR / "label_definition.json").write_text(json.dumps(
        {"min_tmax_c": c.MIN_TMAX_C, "max_discharge_m3s": c.MAX_DISCHARGE_M3S,
         "train": f"{c.HISTORY_START}..{c.TRAIN_END}", "test": "2025-01-01..heute"}, indent=2))

    mr = project.get_model_registry()
    hs_model = mr.python.create_model(
        name=c.MODEL_NAME,
        metrics=metrics,
        description="RandomForest: Ist morgen ein guter Rheinschwimm-Tag in Basel?",
        input_example=X_train.head(1),
        feature_view=fv,  # Lineage: Modell <-> Feature View / Training Dataset
    )
    hs_model.save(str(MODEL_DIR))
    print(f"Modell '{c.MODEL_NAME}' v{hs_model.version} in der Registry gespeichert.")


if __name__ == "__main__":
    main()
