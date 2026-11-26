"""Artifact-backed, seven-page Streamlit analytics application."""

import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from src.config import RAW_FILE, DATA, REPORTS, ARTIFACTS, settings
from src.data.schema import (
    FEATURES,
    NUMERIC,
    CODES,
    COUNTS,
    MONTHS,
    VISITORS,
    validate_frame,
    DataValidationError,
)
from src.data.analytics import analytics, statistical_tests
from src.models.inference import get_predictor
from src.evaluation.metrics import evaluate, threshold_table
from src.monitoring.drift import monitoring_history, simulate, rejection_summary

PAGES = [
    "Executive Overview",
    "Exploratory Data Analysis",
    "Model Performance",
    "Conversion Prediction",
    "Batch Prediction",
    "Model Monitoring",
    "Experiment History",
]


@st.cache_data
def read_data(path, modified):
    return pd.read_csv(path)


def csv(path):
    return read_data(str(path), path.stat().st_mtime_ns)


def chart(figure):
    figure.update_layout(
        template="plotly_white",
        colorway=["#286b78", "#be8252", "#7d8a9b"],
        margin=dict(l=12, r=12, t=40, b=15),
        font=dict(family="Arial", size=13),
    )
    st.plotly_chart(figure, width="stretch")


def overview(data, model):
    result = analytics(data)
    k = result["kpis"]
    columns = st.columns(4)
    for col, label, value in zip(
        columns,
        ["Shopping sessions", "Purchases", "Conversion rate", "Average duration"],
        [
            f"{k['sessions']:,.0f}",
            f"{k['conversions']:,.0f}",
            f"{k['conversion_rate']:.2%}",
            f"{k['average_duration'] / 60:.1f} min",
        ],
    ):
        col.metric(label, value)
    st.caption(
        f"Conversion 95% Wilson interval: {k['conversion_ci'][0]:.2%}–{k['conversion_ci'][1]:.2%}. Historical UCI sessions; not live store activity."
    )
    left, right = st.columns(2)
    with left:
        monthly = result["groups"]["Month"].copy()
        monthly["month_number"] = monthly.segment.map(MONTHS)
        monthly = monthly.sort_values("month_number")
        monthly["error_plus"] = monthly.ci_high - monthly.conversion_rate
        monthly["error_minus"] = monthly.conversion_rate - monthly.ci_low
        chart(
            px.line(
                monthly,
                x="segment",
                y="conversion_rate",
                markers=True,
                error_y="error_plus",
                error_y_minus="error_minus",
                title="Conversion across observed months",
            )
        )
    with right:
        chart(
            px.bar(
                result["groups"]["VisitorType"],
                x="segment",
                y="conversion_rate",
                title="Visitor conversion rates",
                hover_data=["sessions", "ci_low", "ci_high"],
            )
        )
    metrics = st.columns(4)
    metrics[0].metric("New visitor conversion", f"{k['New_Visitor_conversion_rate']:.2%}")
    metrics[1].metric("Returning conversion", f"{k['Returning_Visitor_conversion_rate']:.2%}")
    metrics[2].metric("Mean bounce rate", f"{k['average_bounce_rate']:.2%}")
    metrics[3].metric("Mean exit rate", f"{k['average_exit_rate']:.2%}")
    findings = REPORTS / "eda/findings.json"
    if findings.exists():
        st.subheader("Observed findings")
        for observation in json.loads(findings.read_text())["observations"]:
            st.write("• " + observation)
    if model:
        predictions = pd.DataFrame(model.predict(data[FEATURES]))
        segments = (
            predictions.propensity_segment.value_counts().rename_axis("segment").reset_index(name="sessions")
        )
        chart(
            px.bar(
                segments,
                x="segment",
                y="sessions",
                title="Model propensity segments · includes training sessions",
            )
        )
    st.info(
        "These patterns describe associations. Use controlled experiments before attributing changes in conversion to an intervention."
    )


def eda(data):
    first, second = st.columns(2)
    months = first.multiselect("Months", list(MONTHS), default=sorted(data.Month.unique(), key=MONTHS.get))
    visitors = second.multiselect("Visitor types", VISITORS, default=VISITORS)
    filtered = data[data.Month.isin(months) & data.VisitorType.isin(visitors)]
    if filtered.empty:
        st.info("No sessions match these filters.")
        return
    st.caption(f"{len(filtered):,} matching sessions")
    feature = st.selectbox("Engagement feature", NUMERIC)
    chart(
        px.histogram(
            filtered.assign(Revenue=filtered.Revenue.astype(str)),
            x=feature,
            color="Revenue",
            nbins=40,
            barmode="overlay",
            histnorm="probability density",
            title="Feature distribution by conversion",
        )
    )
    chart(px.box(filtered, x="Revenue", y=feature, title="Engagement distribution", points=False))
    chart(
        px.imshow(
            filtered[NUMERIC + ["Revenue"]].corr(),
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            title="Pearson correlations",
        )
    )
    groups = analytics(filtered)["groups"]
    group = st.selectbox("Compare segments", ["VisitorType", "TrafficType", "Month", "Weekend", "SpecialDay"])
    chart(
        px.bar(
            groups[group],
            x="segment",
            y="conversion_rate",
            hover_data=["sessions", "ci_low", "ci_high"],
            title=f"Conversion by {group}",
        )
    )
    st.dataframe(groups[group], hide_index=True, width="stretch")
    with st.expander("Statistical tests and uncertainty"):
        st.dataframe(statistical_tests(filtered), hide_index=True)
        st.caption(
            "Exploratory tests with Bonferroni-adjusted p-values and effect sizes. Small expected cell counts weaken chi-square approximations; significance does not imply causality."
        )
    with st.expander("Data quality"):
        from src.data.schema import quality_report

        st.json(quality_report(filtered))


def performance(model):
    directory = model.run_dir / "reports"
    st.caption(
        "Algorithm comparison uses validation at threshold 0.5. Only the selected model was evaluated on the test partition."
    )
    comparison = csv(directory / "model_comparison.csv")
    st.dataframe(comparison, hide_index=True, width="stretch")
    chart(
        px.bar(
            comparison,
            x="average_precision",
            y="model",
            orientation="h",
            title="Validation Average Precision",
            color="engineered",
            height=580,
        )
    )
    ablation = comparison.pivot(index="family", columns="engineered", values="average_precision").rename(
        columns={False: "original", True: "engineered"}
    )
    if "engineered" in ablation:
        ablation["AP_change"] = ablation.engineered - ablation.original
    st.subheader("Feature-engineering ablation")
    st.dataframe(ablation)
    selection_path = directory / "feature_selection.json"
    if selection_path.exists():
        with st.expander("Train-fitted feature screening experiment"):
            st.json(json.loads(selection_path.read_text()))
    st.subheader("Final held-out test results")
    st.json(model.metadata["test_metrics"], expanded=False)
    test = csv(directory / "test_predictions.csv")
    left, right = st.columns(2)
    with left:
        fpr, tpr, _ = roc_curve(test.Revenue, test.probability)
        chart(
            px.area(
                x=fpr,
                y=tpr,
                labels={"x": "False positive rate", "y": "True positive rate"},
                title="Test ROC curve",
            )
        )
    with right:
        precision, recall, _ = precision_recall_curve(test.Revenue, test.probability)
        chart(
            px.line(
                x=recall,
                y=precision,
                labels={"x": "Recall", "y": "Precision"},
                title="Test Precision–Recall curve",
            )
        )
    fraction, mean_probability = calibration_curve(
        test.Revenue, test.probability, n_bins=10, strategy="quantile"
    )
    fig = go.Figure(
        [
            go.Scatter(x=mean_probability, y=fraction, mode="lines+markers", name="Observed"),
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(dash="dash")),
        ]
    )
    fig.update_layout(
        title="Test calibration", xaxis_title="Mean predicted probability", yaxis_title="Observed conversion"
    )
    chart(fig)
    st.subheader("Validation decision-policy simulator")
    st.caption(
        "Exploration below uses validation data only. Costs are assumptions in arbitrary currency units; no revenue uplift has been experimentally verified."
    )
    threshold = st.slider("Decision threshold", 0.0, 1.0, float(model.metadata["threshold"]), 0.01)
    left, right = st.columns(2)
    fp_cost = left.number_input("Cost per false positive", min_value=0.0, value=1.0)
    fn_cost = right.number_input("Cost per false negative", min_value=0.0, value=5.0)
    val = csv(directory / "validation_predictions.csv")
    results = evaluate(val.Revenue, val.probability, threshold)
    tn, fp, fn, tp = np.asarray(results["confusion_matrix"]).ravel()
    columns = st.columns(4)
    columns[0].metric("Precision", f"{results['precision']:.2%}")
    columns[1].metric("Recall", f"{results['recall']:.2%}")
    columns[2].metric("F1", f"{results['f1']:.3f}")
    columns[3].metric("Illustrative cost", f"{fp * fp_cost + fn * fn_cost:,.2f}")
    chart(
        px.imshow(
            results["confusion_matrix"],
            x=["Predicted no", "Predicted yes"],
            y=["Actual no", "Actual yes"],
            text_auto=True,
            title="Validation confusion matrix",
        )
    )
    table = pd.DataFrame(threshold_table(val.Revenue, val.probability, fp_cost, fn_cost))
    chart(px.line(table, x="threshold", y=["precision", "recall", "f1"], title="Threshold trade-offs"))
    chart(px.line(table, x="threshold", y="illustrative_cost", title="Assumption-based cost curve"))
    st.subheader("Model explanations")
    importance = csv(directory / "permutation_importance.csv")
    chart(
        px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            error_x="std",
            orientation="h",
            title="Validation permutation importance (AP decrease)",
        )
    )
    for name in ["shap_summary.png", "shap_waterfall.png"]:
        if (directory / name).exists():
            st.image(str(directory / name))
    st.caption(
        "Correlated inputs can share or obscure importance. SHAP contributions explain the fitted model, not the effect of changing customer behavior."
    )


def prediction(model):
    st.caption(
        "Enter complete session aggregates. Defaults are an actual historical example from the training reference."
    )
    example = model.background.iloc[0]
    with st.form("session"):
        fields = {}
        columns = st.columns(3)
        for index, name in enumerate(FEATURES):
            with columns[index % 3]:
                if name == "Month":
                    fields[name] = st.selectbox(name, list(MONTHS), index=list(MONTHS).index(example[name]))
                elif name == "VisitorType":
                    fields[name] = st.selectbox(name, VISITORS, index=VISITORS.index(example[name]))
                elif name == "Weekend":
                    fields[name] = st.checkbox(name, value=bool(example[name]))
                elif name in COUNTS + CODES:
                    fields[name] = st.number_input(
                        name, min_value=1 if name in CODES else 0, value=int(example[name]), step=1
                    )
                else:
                    fields[name] = st.number_input(
                        name,
                        min_value=0.0,
                        max_value=1.0 if name in ["BounceRates", "ExitRates", "SpecialDay"] else None,
                        value=float(example[name]),
                        format="%.5f",
                    )
        threshold = st.slider("Decision threshold", 0.0, 1.0, float(model.metadata["threshold"]), 0.01)
        left, right = st.columns(2)
        medium = left.number_input(
            "Medium intent starts at", 0.0, 1.0, float(model.metadata["segments"]["medium"])
        )
        high = right.number_input(
            "High intent starts at", 0.0, 1.0, float(model.metadata["segments"]["high"])
        )
        submitted = st.form_submit_button("Estimate conversion", type="primary")
    if submitted:
        try:
            frame = pd.DataFrame([fields])
            result = model.predict(frame, threshold=threshold, segments={"medium": medium, "high": high})[0]
            st.metric("Purchase probability", f"{result['conversion_probability']:.1%}")
            st.write(
                f"{result['propensity_segment'].title()} intent · predicted class {result['predicted_class']}"
            )
            st.caption(f"Model {result['model_version']} · threshold {result['decision_threshold']:.2f}")
            with st.spinner("Computing individual explanation…"):
                explanation = model.explain(frame)
            contributions = pd.DataFrame(
                explanation["contributions"].items(), columns=["feature", "contribution"]
            ).sort_values("contribution")
            chart(
                px.bar(
                    contributions,
                    x="contribution",
                    y="feature",
                    orientation="h",
                    title="Features moving the model estimate up or down",
                )
            )
            st.write(explanation["note"])
            st.caption(
                f"Method: {explanation['method']}; units: {explanation['units']}; reference: {explanation['base_value']}"
            )
        except (DataValidationError, ValueError) as exc:
            st.error(str(exc))


def batch(model):
    st.write(
        "Upload up to 10,000 complete sessions. Optional Revenue labels are used only for monitoring; PageValues is ignored by prediction."
    )
    st.download_button(
        "Download valid sample CSV",
        model.background.head(5).to_csv(index=False),
        "convertiq_sample.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Session CSV", type=["csv"], key="batch_csv")
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(uploaded)
        if len(frame) > 10000:
            st.error("Batch limit is 10,000 rows.")
            return
        validate_frame(frame)
        st.dataframe(frame.head(20), hide_index=True)
        if st.button("Generate predictions", type="primary"):
            with st.spinner("Scoring and checking batch quality…"):
                output = pd.concat(
                    [frame.reset_index(drop=True), pd.DataFrame(model.predict(frame, record=True))], axis=1
                )
            st.session_state["batch_result"] = (uploaded.getvalue(), model.metadata["version"], output)
        cached = st.session_state.get("batch_result")
        if cached and cached[0] == uploaded.getvalue() and cached[1] == model.metadata["version"]:
            st.dataframe(cached[2], hide_index=True)
            st.download_button(
                "Download predictions", cached[2].to_csv(index=False), "convertiq_predictions.csv", "text/csv"
            )
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f"Invalid CSV: {exc}")


def monitoring(model):
    st.caption(
        "Reference: training inputs. Historical replays and controlled synthetic drift are explicitly labeled simulations."
    )
    cfg = settings()["monitoring"].copy()
    with st.expander("Alert settings"):
        cfg["psi_alert"] = st.number_input("PSI alert threshold", 0.01, 10.0, float(cfg["psi_alert"]))
        cfg["js_alert"] = st.number_input(
            "Jensen–Shannon divergence alert threshold", 0.01, 1.0, float(cfg["js_alert"])
        )
        cfg["minimum_batch"] = st.number_input(
            "Minimum distribution-monitoring batch", 20, 10000, int(cfg["minimum_batch"])
        )
        cfg["latency_ms_alert"] = st.number_input(
            "Batch latency alert (ms)", 1.0, 60000.0, float(cfg["latency_ms_alert"])
        )
    cfg["minimum_average_precision"] = st.number_input(
        "Labeled batch minimum AP", 0.0, 1.0, float(cfg["minimum_average_precision"])
    )
    cfg["minimum_recall"] = st.number_input(
        "Labeled batch minimum recall", 0.0, 1.0, float(cfg["minimum_recall"])
    )
    rejected = rejection_summary()
    st.metric("Rejected API requests", rejected["count"])
    if rejected["recent"]:
        with st.expander("Recent API validation errors (no payloads stored)"):
            st.json(rejected["recent"])
    source = st.radio("Monitoring input", ["Held-out historical replay", "Uploaded CSV"], horizontal=True)
    frame = None
    synthetic = False
    if source == "Held-out historical replay":
        path = DATA / "processed/test.csv"
        if path.exists():
            frame = csv(path).copy()
            multiplier = st.slider(
                "Synthetic drift multiplier (1.0 keeps historical data)", 1.0, 5.0, 1.0, 0.25
            )
            synthetic = multiplier != 1
            frame = simulate(frame, multiplier)
            if synthetic:
                st.warning(
                    "Synthetic covariate drift: durations and exit rates changed; original labels removed."
                )
    else:
        uploaded = st.file_uploader("Monitoring batch CSV", type=["csv"], key="monitor_csv")
        if uploaded:
            try:
                frame = pd.read_csv(uploaded)
            except ValueError as exc:
                st.error(str(exc))
    if frame is not None and st.button("Analyze monitoring batch", type="primary"):
        if len(frame) > 10000:
            st.error("Batch limit is 10,000 rows.")
        else:
            with st.spinner("Checking data and distributions…"):
                probabilities, latency = None, None
                try:
                    start = time.perf_counter()
                    predictions = model.predict(frame)
                    probabilities = [row["conversion_probability"] for row in predictions]
                    latency = (time.perf_counter() - start) * 1000
                except DataValidationError:
                    pass  # Quality-only report is deliberately available for invalid batches.
                report = model.monitor(
                    frame,
                    probabilities,
                    latency,
                    simulated=source == "Held-out historical replay",
                    config=cfg,
                )
                st.session_state["monitor_report"] = report
    if "monitor_report" in st.session_state:
        report = st.session_state["monitor_report"]
        st.write(f"Batch status: **{report['status']}**")
        for alert in report["alerts"]:
            st.warning(alert)
        st.json(report)
        if report["numeric_psi"]:
            table = pd.DataFrame(report["numeric_psi"].items(), columns=["feature", "PSI"])
            chart(px.bar(table, x="feature", y="PSI", title="Numerical feature drift"))
    history = monitoring_history()
    if history:
        st.subheader("Monitoring history")
        table = pd.DataFrame(
            [
                {
                    "time": r["created_at"],
                    "version": r["model_version"],
                    "simulated": r["simulated"],
                    "rows": r["quality"]["rows"],
                    "status": r["status"],
                    "latency_ms": r["latency_ms"],
                    "average_precision": (r["performance"] or {}).get("average_precision"),
                }
                for r in history
            ]
        )
        st.dataframe(table, hide_index=True)
        labeled = table.dropna(subset=["average_precision"])
        if len(labeled):
            chart(
                px.line(
                    labeled.sort_values("time"),
                    x="time",
                    y="average_precision",
                    color="version",
                    markers=True,
                    title="Performance on batches with known outcomes",
                )
            )
    else:
        st.info("No monitoring batches have been recorded yet.")


def experiments():
    paths = sorted((ARTIFACTS / "runs").glob("*/metadata.json"), reverse=True)
    if not paths:
        st.info("Run training to record experiments.")
        return
    version = st.selectbox("Training run", [path.parent.name for path in paths])
    run_dir = ARTIFACTS / "runs" / version
    st.json(json.loads((run_dir / "metadata.json").read_text()), expanded=False)
    comparison = csv(run_dir / "reports/model_comparison.csv")
    st.dataframe(comparison, hide_index=True, width="stretch")
    experiment = st.selectbox("Inspect experiment parameters", comparison.model.tolist())
    st.json(json.loads((run_dir / "reports" / f"{experiment}.json").read_text()))
    st.caption(
        "Full runs and artifacts are also logged to MLflow. Start the local tracking UI with make mlflow."
    )


def main():
    st.set_page_config(page_title="ConvertIQ", page_icon="C", layout="wide")
    st.markdown(
        """<style>.block-container{max-width:1400px;padding-top:2rem} [data-testid="stMetric"]{background:#f3f6f7;border-radius:8px;padding:16px} h1{letter-spacing:-1px} </style>""",
        unsafe_allow_html=True,
    )
    st.sidebar.title("ConvertIQ")
    st.sidebar.caption("Conversion intelligence")
    page = st.sidebar.radio("Workspace", PAGES, label_visibility="collapsed")
    st.title(page)
    st.caption("Session-level propensity · evidence-led customer analytics")
    model = None
    try:
        model = get_predictor()
        st.sidebar.caption(f"Model: {model.metadata['selected_model']}\n\n{model.metadata['version']}")
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.sidebar.info(str(exc))
    if page in ["Executive Overview", "Exploratory Data Analysis"]:
        if not RAW_FILE.exists():
            st.info("Download the dataset with: python -m src.data.download")
            return
        data = validate_frame(csv(RAW_FILE), target=True, raw=True)
        overview(data, model) if page == "Executive Overview" else eda(data)
    elif page == "Experiment History":
        experiments()
    elif model is None:
        st.info("Train the model with: python -m src.models.train")
    else:
        {
            "Model Performance": performance,
            "Conversion Prediction": prediction,
            "Batch Prediction": batch,
            "Model Monitoring": monitoring,
        }[page](model)
