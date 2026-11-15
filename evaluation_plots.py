"""Evaluation & Visualisierung: Feature Store + Model Registry -> Plots in docs/.

Liest alle Features aus der Feature Group, lädt das beste Modell aus der Registry und erstellt:
  1. docs/saison_2025.png          Rheinschwimm-Saison 2025: Abfluss, Tmax und Ampel im Zeitverlauf
  2. docs/schwimmtage_heatmap.png   Anzahl Rheinschwimm-Tage pro Monat und Jahr (2015-2026)
  3. docs/modell_evaluation.png     Feature Importance + Metriken ganzes Jahr vs. nur Sommer
sowie docs/metrics.json.

Beispiel:
    uv run python evaluation_plots.py
"""

import json
from pathlib import Path

import hopsworks
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402

from ampel import config as c  # noqa: E402
from ampel.openmeteo import fetch_catchment_weather  # noqa: E402

DOCS = Path("docs")

# Farben (Referenzpalette: Surface, Ink, Kategorien, Status)
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e3"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
BLUE_RAMP = ["#f0f5fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MONTHS_DE = {5: "Mai", 6: "Juni", 7: "Juli", 8: "August", 9: "September"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlelocation": "left",
})


def ampel_color(p: float) -> str:
    return GOOD if p >= c.GREEN_PROBA else WARNING if p >= c.YELLOW_PROBA else CRITICAL


def load_data():
    load_dotenv(".env")
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(c.FEATURE_GROUP_NAME, version=c.FEATURE_GROUP_VERSION)
    df = fg.read().sort_values("date_id").reset_index(drop=True)
    df["day"] = pd.to_datetime(df["date_id"])

    hs_model = project.get_model_registry().get_best_model(c.MODEL_NAME, "roc_auc", "max")
    model = joblib.load(Path(hs_model.download()) / "model.pkl")
    df["proba"] = model.predict_proba(df[c.FEATURES])[:, 1]
    return df, model, hs_model.version


def plot_season(df: pd.DataFrame) -> None:
    """Saison 2025 – die Prediction von Tag d gilt für Tag d+1, deshalb wird alles auf den Zieltag geplottet."""
    s = df[["day", "proba", c.LABEL]].copy()
    s["target_day"] = s["day"] + pd.Timedelta(days=1)
    s = s[(s["target_day"] >= "2025-05-01") & (s["target_day"] <= "2025-09-30")]

    q = df.set_index("day")["discharge_m3s"]
    tmax = fetch_catchment_weather("2025-05-01", "2025-09-30").set_index("date")["tmax_basel_c"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.6], "hspace": 0.35})
    fig.suptitle("Rheinschwimm-Saison 2025 – was das Modell am Vortag vorhergesagt hat",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")

    ax = axes[0]
    q25 = q["2025-05-01":"2025-09-30"]
    ax.fill_between(q25.index, c.MAX_DISCHARGE_M3S, q25.values, where=q25.values > c.MAX_DISCHARGE_M3S,
                    color=CRITICAL, alpha=0.15, linewidth=0, interpolate=True)
    ax.plot(q25.index, q25.values, color=BLUE, linewidth=1.8)
    ax.axhline(c.MAX_DISCHARGE_M3S, color=INK_2, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(pd.Timestamp("2025-06-20"), c.MAX_DISCHARGE_M3S + 50, f"Grenze {c.MAX_DISCHARGE_M3S:,.0f} m³/s".replace(",", "'"),
            ha="center", va="bottom", color=INK_2, fontsize=9,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_title("Abfluss Rhein Basel (m³/s)")

    ax = axes[1]
    ax.fill_between(tmax.index, c.MIN_TMAX_C, tmax.values, where=tmax.values >= c.MIN_TMAX_C,
                    color=GOOD, alpha=0.15, linewidth=0, interpolate=True)
    ax.plot(tmax.index, tmax.values, color=ORANGE, linewidth=1.8)
    ax.axhline(c.MIN_TMAX_C, color=INK_2, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(pd.Timestamp("2025-09-24"), c.MIN_TMAX_C + 4.5, f"Schwelle {c.MIN_TMAX_C:.0f} °C",
            ha="center", va="bottom", color=INK_2, fontsize=9,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_title("Tageshöchsttemperatur Basel (°C)")

    ax = axes[2]
    ax.grid(False)
    ax.bar(s["target_day"], 1, width=0.85, bottom=0.55, color=[ampel_color(p) for p in s["proba"]])
    hits = s[s[c.LABEL] == 1]
    ax.scatter(hits["target_day"], np.full(len(hits), 0.2), marker="o", s=14, color=INK, zorder=3)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([0.2, 1.05], ["tatsächlich\nSchwimmtag", "Ampel\n(Vortag)"])
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=INK_2)
    ax.spines["left"].set_visible(False)
    ax.set_title("Prognose vs. Realität")
    ax.legend(handles=[Patch(color=GOOD, label="grün ≥ 60%"),
                       Patch(color=WARNING, label="gelb 30–60%"),
                       Patch(color=CRITICAL, label="rot < 30%"),
                       plt.Line2D([], [], marker="o", linestyle="", color=INK, markersize=4,
                                  label="tatsächlicher Schwimmtag")],
              loc="upper left", bbox_to_anchor=(0, -0.35), ncol=4, frameon=False, fontsize=9)

    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    de = {1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez"}
    axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{de[mdates.num2date(v).month]} {mdates.num2date(v).year}"))
    for a in axes[:2]:
        a.spines["left"].set_color(GRID)
    fig.savefig(DOCS / "saison_2025.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame) -> None:
    """Anzahl tatsächlicher Rheinschwimm-Tage (Label gilt für den Folgetag) pro Monat und Jahr."""
    s = df.dropna(subset=[c.LABEL]).copy()
    s["target_day"] = s["day"] + pd.Timedelta(days=1)
    s = s[s["target_day"].dt.month.isin(MONTHS_DE)]
    pivot = (s.assign(year=s["target_day"].dt.year, month=s["target_day"].dt.month)
              .pivot_table(index="year", columns="month", values=c.LABEL, aggfunc="sum")
              .reindex(columns=list(MONTHS_DE)).fillna(0).astype(int))
    totals = pivot.sum(axis=1)

    fig, (ax, ax_tot) = plt.subplots(1, 2, figsize=(10, 6.2), gridspec_kw={"width_ratios": [5, 1.6], "wspace": 0.08})
    fig.suptitle("Wie viele Rheinschwimm-Tage hatte Basel? (≥ 24 °C und ≤ 1'800 m³/s)",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    vmax = max(int(pivot.values.max()), 1)
    ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(pivot.values):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                color="#ffffff" if v > vmax * 0.55 else INK)
    ax.set_xticks(range(len(MONTHS_DE)), list(MONTHS_DE.values()))
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(MONTHS_DE)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    ax_tot.barh(range(len(totals)), totals.values, color=BLUE, height=0.7)
    ax_tot.set_ylim(len(totals) - 0.5, -0.5)
    ax_tot.set_yticks([])
    ax_tot.set_title("Total", fontsize=10)
    ax_tot.grid(axis="y", visible=False)
    ax_tot.spines["left"].set_visible(False)
    for i, v in enumerate(totals.values):
        ax_tot.text(v + 1, i, str(v), va="center", fontsize=9, color=INK_2)
    ax_tot.set_xlim(0, totals.max() * 1.3)
    ax_tot.tick_params(axis="x", labelsize=8)
    fig.text(0.06, 0.01, "Trainingsdaten 2015–2024, Test 2025–2026. Quelle: Open-Meteo (Historical Weather + GloFAS Flood API).",
             fontsize=8, color=MUTED)
    fig.savefig(DOCS / "schwimmtage_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate(df: pd.DataFrame) -> dict:
    test = df[(df["day"] >= "2025-01-01")].dropna(subset=[c.LABEL]).copy()
    test["y"] = test[c.LABEL].astype(int)
    summer = test[(test["day"] + pd.Timedelta(days=1)).dt.month.isin([6, 7, 8, 9])]

    def m(d):
        pred = (d["proba"] >= 0.5).astype(int)
        return {"roc_auc": round(roc_auc_score(d["y"], d["proba"]), 3),
                "f1": round(f1_score(d["y"], pred), 3),
                "precision": round(precision_score(d["y"], pred), 3),
                "recall": round(recall_score(d["y"], pred), 3),
                "n_days": int(len(d)), "n_swim_days": int(d["y"].sum())}
    return {"test_ganzes_jahr": m(test), "test_nur_sommer_jun_sep": m(summer)}


def plot_model(model, metrics: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.4})
    fig.suptitle("Modell-Evaluation (Out-of-Sample 2025–2026)", x=0.06, ha="left", fontsize=13, fontweight="bold")

    names = {"temp_now_c": "Temperatur jetzt (RT)", "tmax_basel_3d_mean_c": "Tmax Ø 3 Tage",
             "day_of_year": "Tag im Jahr", "discharge_m3s": "Abfluss heute",
             "cloud_cover_now_pct": "Bewölkung jetzt (RT)", "precip_catchment_3d_mm": "Regen Einzugsgebiet 3T",
             "discharge_trend_3d_m3s": "Abfluss-Trend 3T"}
    imp = pd.Series(model.feature_importances_, index=c.FEATURES).sort_values()
    ax1.barh([names[f] for f in imp.index], imp.values, color=BLUE, height=0.6)
    for i, v in enumerate(imp.values):
        ax1.text(v + 0.008, i, f"{v:.2f}", va="center", fontsize=9, color=INK_2)
    ax1.set_title("Feature Importance (Random Forest)")
    ax1.grid(axis="y", visible=False)
    ax1.set_xlim(0, imp.max() * 1.2)
    ax1.tick_params(axis="y", length=0, labelcolor=INK)

    keys = ["roc_auc", "f1", "precision", "recall"]
    labels = ["ROC-AUC", "F1", "Precision", "Recall"]
    x = np.arange(len(keys))
    w = 0.36
    a = [metrics["test_ganzes_jahr"][k] for k in keys]
    b = [metrics["test_nur_sommer_jun_sep"][k] for k in keys]
    ax2.bar(x - w / 2 - 0.01, a, w, color=BLUE, label="ganzes Jahr")
    ax2.bar(x + w / 2 + 0.01, b, w, color=ORANGE, label="nur Sommer (Jun–Sep)")
    for xi, (va, vb) in enumerate(zip(a, b)):
        ax2.text(xi - w / 2 - 0.01, va + 0.015, f"{va:.2f}", ha="center", fontsize=8, color=INK_2)
        ax2.text(xi + w / 2 + 0.01, vb + 0.015, f"{vb:.2f}", ha="center", fontsize=8, color=INK_2)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 1.12)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Metriken: ganzes Jahr vs. nur Sommer")
    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, fontsize=9)
    fig.savefig(DOCS / "modell_evaluation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    df, model, version = load_data()
    metrics = evaluate(df)
    metrics["model_version"] = int(version)
    (DOCS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    plot_season(df)
    plot_heatmap(df)
    plot_model(model, metrics)
    print(f"Plots gespeichert in {DOCS.resolve()}")


if __name__ == "__main__":
    main()
