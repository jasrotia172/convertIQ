"""DuckDB business analytics, uncertainty estimates and exploratory charts."""

import math
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from src.config import RAW_FILE, REPORTS, save_json
from src.data.schema import validate_frame, NUMERIC, DURATIONS, MONTHS


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return center - half, center + half


def analytics(frame: pd.DataFrame) -> dict:
    connection = duckdb.connect(":memory:")
    connection.register("sessions", frame)
    kpis = (
        connection.execute("""SELECT COUNT(*) AS sessions, SUM(CAST(Revenue AS INT)) AS conversions,
        AVG(CAST(Revenue AS INT)) AS conversion_rate,
        1-AVG(CAST(Revenue AS INT)) AS non_conversion_rate,
        AVG(Administrative_Duration+Informational_Duration+ProductRelated_Duration) AS average_duration,
        AVG(BounceRates) AS average_bounce_rate, AVG(ExitRates) AS average_exit_rate FROM sessions""")
        .df()
        .iloc[0]
        .to_dict()
    )
    groups = {}
    for column in ["Month", "TrafficType", "VisitorType", "Weekend", "SpecialDay"]:
        grouped = connection.execute(f'''SELECT "{column}" AS segment, COUNT(*) AS sessions,
            SUM(CAST(Revenue AS INT)) AS conversions, AVG(CAST(Revenue AS INT)) AS conversion_rate
            FROM sessions GROUP BY "{column}" ORDER BY sessions DESC''').df()
        intervals = [wilson(int(row.conversions), int(row.sessions)) for row in grouped.itertuples()]
        grouped["ci_low"] = [x[0] for x in intervals]
        grouped["ci_high"] = [x[1] for x in intervals]
        groups[column] = grouped
    connection.close()
    kpis["conversion_ci"] = wilson(int(kpis["conversions"]), int(kpis["sessions"]))
    for kind in ["Returning_Visitor", "New_Visitor"]:
        selection = groups["VisitorType"].query("segment == @kind")
        kpis[f"{kind}_conversion_rate"] = float(selection.conversion_rate.iloc[0]) if len(selection) else None
    return {"kpis": kpis, "groups": groups}


def statistical_tests(frame):
    rows = []
    for column in ["VisitorType", "Month", "TrafficType", "Weekend", "SpecialDay"]:
        table = pd.crosstab(frame[column], frame.Revenue)
        if min(table.shape) < 2:
            continue
        chi2, p, dof, expected = chi2_contingency(table)
        rows.append(
            {
                "feature": column,
                "test": "chi_square",
                "statistic": float(chi2),
                "p_value": float(p),
                "effect_size": float(
                    np.sqrt(chi2 / (table.values.sum() * min(table.shape[0] - 1, table.shape[1] - 1)))
                ),
                "effect_type": "Cramers V",
                "sparse_expected_cells": int((expected < 5).sum()),
            }
        )
    for column in NUMERIC:
        positive = frame.loc[frame.Revenue, column]
        negative = frame.loc[~frame.Revenue, column]
        if len(positive) and len(negative):
            u, p = mannwhitneyu(positive, negative, alternative="two-sided")
            rows.append(
                {
                    "feature": column,
                    "test": "Mann_Whitney_U",
                    "statistic": float(u),
                    "p_value": float(p),
                    "effect_size": float(2 * u / (len(positive) * len(negative)) - 1),
                    "effect_type": "rank_biserial",
                    "sparse_expected_cells": 0,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bonferroni"] = (result.p_value * len(result)).clip(upper=1)
    return result


def generate_report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = validate_frame(pd.read_csv(RAW_FILE), target=True, raw=True)
    result = analytics(frame)
    output = REPORTS / "eda"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "kpis.json", result["kpis"])
    for name, grouped in result["groups"].items():
        grouped.to_csv(output / f"{name}.csv", index=False)
    statistical_tests(frame).to_csv(output / "statistical_tests.csv", index=False)
    frame.describe(include="all").to_csv(output / "descriptive_statistics.csv")
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    monthly = result["groups"]["Month"].copy()
    monthly["order"] = monthly.segment.map(MONTHS)
    monthly = monthly.sort_values("order")
    axes[0].errorbar(
        monthly.segment,
        monthly.conversion_rate,
        yerr=[monthly.conversion_rate - monthly.ci_low, monthly.ci_high - monthly.conversion_rate],
        fmt="o-",
        capsize=3,
    )
    axes[0].set(title="Monthly conversion (95% Wilson intervals)", ylabel="Conversion rate")
    visitors = result["groups"]["VisitorType"]
    axes[1].bar(visitors.segment, visitors.conversion_rate, color="#397b87")
    axes[1].set(title="Conversion by visitor type", ylabel="Conversion rate")
    fig.tight_layout()
    fig.savefig(output / "conversion_segments.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(frame[NUMERIC + ["Revenue"]].corr(), cmap="vlag", center=0, ax=ax, annot=True, fmt=".2f")
    fig.tight_layout()
    fig.savefig(output / "correlation.png", dpi=150)
    plt.close(fig)
    view = frame.copy()
    view["TotalDuration"] = frame[DURATIONS].sum(axis=1)
    view["log1p_duration"] = np.log1p(view.TotalDuration)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(
        view, x="log1p_duration", hue="Revenue", bins=40, stat="density", common_norm=False, ax=axes[0, 0]
    )
    sns.boxplot(view, x="Revenue", y="log1p_duration", ax=axes[0, 1])
    sns.boxplot(view, x="Revenue", y="BounceRates", showfliers=False, ax=axes[1, 0])
    sns.boxplot(view, x="Revenue", y="ExitRates", showfliers=False, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(output / "engagement.png", dpi=150)
    plt.close(fig)
    kpis = result["kpis"]
    traffic = (
        result["groups"]["TrafficType"]
        .query("sessions >= 100")
        .sort_values("conversion_rate", ascending=False)
        .iloc[0]
    )
    notes = [
        f"Observed conversion: {int(kpis['conversions']):,}/{int(kpis['sessions']):,} ({kpis['conversion_rate']:.2%}).",
        f"95% Wilson interval: {kpis['conversion_ci'][0]:.2%}–{kpis['conversion_ci'][1]:.2%}.",
        f"Among traffic codes with at least 100 sessions, code {traffic.segment} has the highest observed rate ({traffic.conversion_rate:.2%}, n={int(traffic.sessions)}). Codes have no channel-name mapping.",
    ]
    for kind in ["Returning_Visitor", "New_Visitor"]:
        notes.append(f"{kind}: {kpis[f'{kind}_conversion_rate']:.2%} observed conversion.")
    save_json(
        output / "findings.json",
        {
            "observations": notes,
            "limitations": "Exploratory associations; repeated tests adjusted with Bonferroni. Sparse chi-square cells weaken asymptotic inference. No causal or contemporary market claims.",
        },
    )
    print("\n".join(notes))


if __name__ == "__main__":
    generate_report()
