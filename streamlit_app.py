
"""GridShift dashboard. Run: streamlit run streamlit_app.py
Requires artifacts from scripts/01..03 and optionally scripts/04.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gridshift import config as C
from gridshift import data as D
from gridshift import metrics as M
from gridshift import models as MD
from gridshift import simulator as S


st.set_page_config(
    page_title="GridShift - ERCOT load forecasting",
    page_icon="⚡",
    layout="wide"
)


# ------------------------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------------------------

def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(obj):
    try:
        st.dataframe(obj, width="stretch")
    except TypeError:
        st.dataframe(obj, use_container_width=True)


# ------------------------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return D.load_clean()


def available_models() -> list[str]:
    names = []

    # Ridge is the selected model and therefore appears first.
    for n, f in [
        ("ridge", "ridge.joblib"),
        ("lightgbm", "lightgbm.joblib"),
        ("cnn_lstm_attn", "cnn_lstm_attn.keras")
    ]:
        if (C.MODELS_DIR / f).exists():
            names.append(n)

    return names


MODEL_LABEL = {
    "ridge": "Ridge (selected model)",
    "lightgbm": "LightGBM (comparison model)",
    "cnn_lstm_attn": "CNN-LSTM-Attention (experimental)"
}


@st.cache_resource(show_spinner="Loading model ...")
def get_forecaster(name: str) -> S.Forecaster:
    df = get_data()

    if name == "cnn_lstm_attn":
        from gridshift import deep
        m, sc = deep.load()
        return S.Forecaster(df, S.deep_predictor(m, sc))

    return S.Forecaster(
        df,
        S.tabular_predictor(MD.load_model(name))
    )


def read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def need_artifacts():
    st.error("Model artifacts not found. Run these from the project root first:")
    st.code(
        "python scripts/01_prepare_data.py\n"
        "python scripts/02_train_models.py\n"
        "python scripts/03_explain.py\n"
        "python scripts/04_explain_ridge.py"
    )
    st.stop()


# ------------------------------------------------------------------------------------
# UI components
# ------------------------------------------------------------------------------------

def pick_origin(fc: S.Forecaster, key: str):
    choices = fc.origin_choices()
    last = choices[-1]

    default_day = (
        last - pd.Timedelta(days=30)
    ).normalize().date()

    c1, c2 = st.columns(2)

    day = c1.date_input(
        "Forecast date (origin)",
        value=default_day,
        min_value=choices[0].date(),
        max_value=last.date(),
        key=f"{key}_d"
    )

    hour = c2.selectbox(
        "Origin hour (load known up to this hour)",
        list(range(24)),
        index=0,
        key=f"{key}_h"
    )

    ts = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))

    if ts not in choices:
        st.warning("That origin is outside the available range; pick another.")
        st.stop()

    return ts


def model_select(key: str) -> str:
    models = available_models()

    if not models:
        need_artifacts()

    return st.selectbox(
        "Model",
        models,
        format_func=lambda n: MODEL_LABEL.get(n, n),
        key=key
    )


# ------------------------------------------------------------------------------------
# Overview page
# ------------------------------------------------------------------------------------

def page_overview():
    st.title("⚡ GridShift")

    st.caption(
        "Weather-aware ERCOT 24-hour load forecasting and "
        "what-if scenario simulation"
    )

    df = get_data()

    c = st.columns(4)

    c[0].metric("Hourly records", f"{len(df):,}")
    c[1].metric("From", str(df.index.min().date()))
    c[2].metric("To", str(df.index.max().date()))
    c[3].metric(
        "Peak load in data (MW)",
        f"{df[C.LOAD_COL].max():,.0f}"
    )

    st.subheader("Test-set accuracy")

    mj = Path(C.REPORTS_DIR / "metrics.json")

    if mj.exists():
        m = json.loads(mj.read_text())

        rows = {
            MODEL_LABEL.get(k, k): v
            for k, v in m.items()
            if isinstance(v, dict)
        }

        t = pd.DataFrame(rows).T

        table(
            t.style.format({
                "MAE": "{:,.0f}",
                "RMSE": "{:,.0f}",
                "MAPE_%": "{:.2f}",
                "R2": "{:.4f}",
                "n": "{:,.0f}",
                "peak_MAE": "{:,.0f}",
                "peak_bias": "{:,.0f}",
                "peak_MAPE_%": "{:.2f}"
            }, na_rep="-")
        )

        st.caption(
            "Computed on the held-out 2024 test split over all "
            "(origin, horizon) rows of OVERLAPPING 24h windows "
            "(a new window every few hours, so each target hour is "
            "predicted several times at different horizons; rows are "
            "not independent). Horizon-wise errors are shown separately "
            "below. Weather for the target hours is the observed weather "
            "(a perfect-forecast assumption), so real-world accuracy with "
            "imperfect weather forecasts would be lower."
        )

        # Ridge is displayed first because it is the selected model.
        for n in ("ridge", "lightgbm", "cnn_lstm_attn"):
            h = read_csv(
                C.REPORTS_DIR / f"metrics_by_horizon_{n}.csv"
            )

            if h is not None:
                st.markdown(
                    f"**Error by forecast horizon - {MODEL_LABEL[n]}**"
                )

                st.line_chart(
                    h.set_index("horizon")[["MAE", "RMSE"]]
                )

    else:
        st.info(
            "No metrics yet. Run `python scripts/02_train_models.py`."
        )

    st.subheader("How it works")

    st.markdown("""
- **Forecast:** Predict the next 24 hours of ERCOT demand using load history, weather (temperature, humidity, wind, precipitation and derived features), and calendar information.
- **Models:** Ridge (selected model), LightGBM (comparison model), and CNN-LSTM-Attention (experimental).
- **Explain:** SHAP, permutation importance, PDP/ALE, weather sensitivity and interactions in Model Insights. LightGBM and Ridge explainability results are identified separately.
- **Simulate:** Modify weather over the next 24 hours and compare baseline versus scenario demand and peak values.
""")


# ------------------------------------------------------------------------------------
# 24-hour forecast page
# ------------------------------------------------------------------------------------

def page_forecast():
    st.title("24-hour demand forecast")

    name = model_select("fm")
    fc = get_forecaster(name)

    origin = pick_origin(fc, "f")

    fut = fc.forecast(origin)

    actual = fc.df[C.LOAD_COL].reindex(fut.index)

    fig = go.Figure()

    fig.add_scatter(
        x=fut.index,
        y=fut["forecast_mw"],
        name="Forecast",
        mode="lines+markers"
    )

    if actual.notna().all():
        fig.add_scatter(
            x=fut.index,
            y=actual,
            name="Actual (replay)",
            mode="lines"
        )

    hist = fc.df[C.LOAD_COL].loc[
        origin - pd.Timedelta(hours=24):origin
    ]

    fig.add_scatter(
        x=hist.index,
        y=hist,
        name="Last 24h actual",
        mode="lines",
        line=dict(dash="dot")
    )

    fig.add_vline(
        x=origin,
        line_dash="dash",
        line_color="gray"
    )

    fig.update_layout(
        yaxis_title="MW",
        height=420,
        legend=dict(orientation="h"),
        margin=dict(t=20)
    )

    show(fig)

    cols = st.columns(4)

    cols[0].metric(
        "Forecast peak (MW)",
        f"{fut['forecast_mw'].max():,.0f}",
        f"at {fut['forecast_mw'].idxmax():%H:%M}"
    )

    if actual.notna().all():
        mm = M.regression_metrics(
            actual,
            fut["forecast_mw"]
        )

        cols[1].metric(
            "Actual peak (MW)",
            f"{actual.max():,.0f}"
        )

        cols[2].metric(
            "MAE (MW)",
            f"{mm['MAE']:,.0f}"
        )

        cols[3].metric(
            "MAPE",
            f"{mm['MAPE_%']:.2f}%"
        )

    st.subheader("Weather used (observed weather as the forecast)")

    st.line_chart(
        fut[["tmpc", "feel"]].rename(
            columns={
                "tmpc": "Temp (°C)",
                "feel": "Feels-like (°F)"
            }
        )
    )

    out = fut.copy()
    out["actual_mw"] = actual

    table(out.round(2))

    st.download_button(
        "Download forecast CSV",
        out.to_csv().encode(),
        "gridshift_forecast.csv",
        "text/csv"
    )


# ------------------------------------------------------------------------------------
# Scenario presets
# ------------------------------------------------------------------------------------

PRESETS = {
    "Custom": None,

    "Heat wave (+6 °C, -10% RH, calm wind)": {
        "t": 6.0,
        "h": -10,
        "w": 0.6,
        "p": 1.0
    },

    "Cold snap (-8 °C, +5% RH, windy)": {
        "t": -8.0,
        "h": 5,
        "w": 1.5,
        "p": 1.0
    },

    "Humid & stormy (+2 °C, +20% RH, wind x1.5, +1 mm/h)": {
        "t": 2.0,
        "h": 20,
        "w": 1.5,
        "p": 1.0,
        "pa": 1.0
    }
}


# ------------------------------------------------------------------------------------
# Scenario simulator page
# ------------------------------------------------------------------------------------

def page_simulator():
    st.title("Weather scenario simulator")

    st.caption(
        "Edit the next-24h weather; derived features (feels-like, CDD/HDD, "
        "trailing 24h extremes, 3h change, 6h precip) are recomputed "
        "automatically. Demand history is unchanged and the model is not "
        "retrained. Differences are MODEL-BASED what-if estimates, not "
        "proven causal effects. Historical-replay mode: the baseline "
        "weather is the observed weather of the next 24 h (no live weather "
        "forecast is used)."
    )

    name = model_select("sm")
    fc = get_forecaster(name)

    origin = pick_origin(fc, "s")

    preset = st.selectbox(
        "Preset",
        list(PRESETS)
    )

    p = PRESETS[preset] or {}

    c1, c2 = st.columns(2)

    dt = c1.slider(
        "Temperature shift (°C)",
        -15.0,
        15.0,
        float(p.get("t", 0.0)),
        0.5
    )

    dh = c1.slider(
        "Humidity shift (percentage points)",
        -50,
        50,
        int(p.get("h", 0)),
        1
    )

    ws = c2.slider(
        "Wind speed multiplier",
        0.0,
        3.0,
        float(p.get("w", 1.0)),
        0.1
    )

    ps = c2.slider(
        "Precipitation multiplier",
        0.0,
        5.0,
        float(p.get("p", 1.0)),
        0.1
    )

    pa = c2.slider(
        "Extra precipitation (mm/h)",
        0.0,
        10.0,
        float(p.get("pa", 0.0)),
        0.25
    )

    hs, he = st.slider(
        "Apply to forecast hours (1 = first hour after origin)",
        1,
        24,
        (1, 24)
    )

    sc = S.Scenario(
        dt,
        dh,
        ws,
        ps,
        pa,
        hs,
        he
    )

    res = fc.compare(origin, sc)

    h = res["hourly"]
    s = res["summary"]

    k = st.columns(4)

    k[0].metric(
        "Baseline peak (MW)",
        f"{s['baseline_peak_mw']:,.0f}",
        f"{s['baseline_peak_time']:%H:%M}",
        delta_color="off"
    )

    k[1].metric(
        "Scenario peak (MW)",
        f"{s['scenario_peak_mw']:,.0f}",
        f"{s['peak_delta_mw']:+,.0f} MW ({s['peak_delta_pct']:+.1f}%)"
    )

    k[2].metric(
        "Peak hour (scenario)",
        f"{s['scenario_peak_time']:%H:%M}"
    )

    k[3].metric(
        "Energy change (MWh / 24h)",
        f"{s['energy_delta_mwh']:+,.0f}"
    )

    if sc.is_identity():
        st.info("Scenario equals baseline (no weather changes).")

    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    fig.add_scatter(
        x=h.index,
        y=h["baseline_mw"],
        name="Baseline forecast",
        mode="lines+markers"
    )

    fig.add_scatter(
        x=h.index,
        y=h["scenario_mw"],
        name="Scenario forecast",
        mode="lines+markers"
    )

    if "actual_mw" in h:
        fig.add_scatter(
            x=h.index,
            y=h["actual_mw"],
            name="Actual (replay)",
            mode="lines",
            line=dict(dash="dot", color="gray")
        )

    fig.add_bar(
        x=h.index,
        y=h["delta_mw"],
        name="Delta MW (right axis)",
        opacity=0.3,
        secondary_y=True
    )

    fig.update_yaxes(
        title_text="MW",
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Delta MW",
        secondary_y=True
    )

    fig.update_layout(
        height=450,
        legend=dict(orientation="h"),
        margin=dict(t=20)
    )

    show(fig)

    st.subheader("Weather: baseline vs scenario")

    wb = res["weather_baseline"]
    ws_ = res["weather_scenario"]

    labels = [
        ("tmpc", "Temperature (°C)"),
        ("feel", "Feels-like (°F)"),
        ("relh", "Humidity (%)"),
        ("sped", "Wind (mph)"),
        ("p01m", "Precip (mm/h)")
    ]

    cols = st.columns(len(labels))

    for col, (v, lab) in zip(cols, labels):
        f = go.Figure()

        f.add_scatter(
            x=wb.index,
            y=wb[v],
            name="Baseline"
        )

        f.add_scatter(
            x=ws_.index,
            y=ws_[v],
            name="Scenario"
        )

        f.update_layout(
            height=230,
            title=dict(text=lab, font=dict(size=13)),
            showlegend=False,
            margin=dict(l=10, r=10, t=35, b=10)
        )

        with col:
            show(f)

    table(h.round(1))

    st.download_button(
        "Download comparison CSV",
        h.to_csv().encode(),
        "gridshift_scenario.csv",
        "text/csv"
    )


# ------------------------------------------------------------------------------------
# Model insights and explainability page
# ------------------------------------------------------------------------------------

def page_insights():
    st.title("Model insights & explainability")

    d = C.EXPLAIN_DIR

    if not d.exists():
        st.info(
            "Run `python scripts/03_explain.py` to generate "
            "LightGBM explainability outputs."
        )
        return

    st.caption(
        "The SHAP, permutation, PDP, ALE, sensitivity and interaction "
        "outputs in the first five tabs describe the LightGBM comparison "
        "model on the 2024 test split (MW), not the selected Ridge model. "
        "Ridge explainability results are presented separately in the "
        "Ridge analysis tab. These are model-attributed or model-based "
        "what-if estimates, not causal effects of weather on demand. "
        "Temperature, feels-like, CDD/HDD and trailing extremes are "
        "strongly correlated: interpret individual-feature scores "
        "carefully and prefer grouped results."
    )

    tabs = st.tabs([
        "SHAP",
        "Permutation importance",
        "PDP / ALE",
        "Weather sensitivity",
        "Interactions",
        "Ridge analysis"
    ])

    # ----------------------------------------------------------------
    # Tab 1: LightGBM SHAP
    # ----------------------------------------------------------------

    with tabs[0]:
        g = read_csv(d / "shap_global.csv")
        grp = read_csv(d / "shap_groups.csv")

        if g is not None:
            top = g.head(20).iloc[::-1]

            show(
                go.Figure(
                    go.Bar(
                        x=top["mean_abs_shap_mw"],
                        y=top["feature"],
                        orientation="h"
                    )
                ).update_layout(
                    title="Mean |SHAP| (MW)",
                    height=550
                )
            )

        if grp is not None:
            show(
                go.Figure(
                    go.Bar(
                        x=grp["group"],
                        y=grp["mean_abs_shap_mw"]
                    )
                ).update_layout(
                    title="Joint mean |SHAP| by feature group (groups overlap; do not add)",
                    height=320
                )
            )

        if (d / "shap_beeswarm.png").exists():
            st.image(
                str(d / "shap_beeswarm.png"),
                caption="LightGBM SHAP summary (beeswarm)"
            )

    # ----------------------------------------------------------------
    # Tab 2: LightGBM permutation importance
    # ----------------------------------------------------------------

    with tabs[1]:
        pg = read_csv(d / "perm_importance_groups.csv")
        pf = read_csv(d / "perm_importance.csv")

        if pg is not None:
            st.markdown(
                "**Grouped permutation importance** - members of each "
                "group are shuffled together. Groups overlap; "
                "'All weather' contains the weather groups."
            )

            show(
                go.Figure(
                    go.Bar(
                        x=pg["name"],
                        y=pg["rmse_increase_mw"],
                        error_y=dict(array=pg["std"])
                    )
                ).update_layout(
                    yaxis_title="RMSE increase (MW)",
                    height=350
                )
            )

        wc = read_csv(
            d / "weather_feature_correlations.csv"
        )

        if wc is not None:
            with st.expander(
                "Correlation between weather features "
                "(why single-feature scores are shared)"
            ):
                table(
                    wc.set_index(wc.columns[0]).round(2)
                )

        if pf is not None:
            top = pf.head(20).iloc[::-1]

            show(
                go.Figure(
                    go.Bar(
                        x=top["rmse_increase_mw"],
                        y=top["name"],
                        orientation="h"
                    )
                ).update_layout(
                    title="LightGBM single-feature permutation importance",
                    xaxis_title="RMSE increase (MW)",
                    height=550
                )
            )

    # ----------------------------------------------------------------
    # Tab 3: PDP and ALE
    # ----------------------------------------------------------------

    with tabs[2]:
        pdp = read_csv(d / "pdp.csv")
        ale = read_csv(d / "ale.csv")

        if pdp is not None:
            st.markdown(
                "**Partial dependence** - variable fixed for the whole "
                "24h window, derived features recomputed."
            )

            for v, g in pdp.groupby("variable"):
                f = go.Figure()

                f.add_scatter(
                    x=g["value"],
                    y=g["peak_demand_mw"],
                    name="Peak"
                )

                f.add_scatter(
                    x=g["value"],
                    y=g["mean_demand_mw"],
                    name="Mean"
                )

                f.update_layout(
                    title=v,
                    height=280,
                    yaxis_title="MW",
                    margin=dict(t=40)
                )

                show(f)

        if ale is not None:
            st.markdown(
                "**ALE** - direct effect of the single raw column; "
                "derived features (feels-like, CDD/HDD, etc.) are held "
                "fixed, so effects that flow through them are not included."
            )

            for v, g in ale.groupby("feature"):
                show(
                    go.Figure(
                        go.Scatter(
                            x=g["x"],
                            y=g["ale_mw"]
                        )
                    ).update_layout(
                        title=f"ALE {v}",
                        height=260,
                        margin=dict(t=40)
                    )
                )

    # ----------------------------------------------------------------
    # Tab 4: Weather sensitivity
    # ----------------------------------------------------------------

    with tabs[3]:
        sen = read_csv(d / "sensitivity.csv")

        if sen is not None:
            season = st.selectbox(
                "Season",
                ["All", "Winter", "Spring", "Summer", "Fall"]
            )

            t = sen[
                sen["season"] == season
            ][
                [
                    "variable",
                    "perturbation",
                    "n_origins",
                    "peak_delta_mw_mean",
                    "peak_delta_mw_std",
                    "mean_delta_mw_mean"
                ]
            ]

            table(t.round(1))

            st.caption(
                "Model-based what-if estimates: change in forecast peak "
                "and mean demand versus the model's baseline forecast, "
                "averaged over test-period origins. Not proven causal effects."
            )

    # ----------------------------------------------------------------
    # Tab 5: Interactions
    # ----------------------------------------------------------------

    with tabs[4]:
        ig = read_csv(d / "interaction_grid.csv")
        si = read_csv(d / "shap_interactions_weather.csv")

        if ig is not None:
            z = ig.pivot(
                index="humidity_delta_pct",
                columns="temp_delta_c",
                values="peak_delta_mw"
            )

            show(
                go.Figure(
                    go.Heatmap(
                        z=z.values,
                        x=z.columns,
                        y=z.index,
                        colorbar=dict(title="Delta peak MW")
                    )
                ).update_layout(
                    title=(
                        "Peak change: temperature shift (°C) x "
                        "humidity shift (pp), summer origins"
                    ),
                    xaxis_title="Temp shift (°C)",
                    yaxis_title="Humidity shift (pp)",
                    height=420
                )
            )

        if si is not None:
            si = si.set_index(si.columns[0])

            show(
                go.Figure(
                    go.Heatmap(
                        z=si.values,
                        x=si.columns,
                        y=si.index
                    )
                ).update_layout(
                    title="Mean |SHAP interaction| (MW)",
                    height=450
                )
            )

    # ----------------------------------------------------------------
    # Tab 6: Ridge analysis
    # ----------------------------------------------------------------

    with tabs[5]:
        st.subheader("Ridge selected-model explainability")

        ridge_dir = ROOT / "artifacts" / "explain_ridge"

        st.caption(
            "These results describe the selected Ridge model. "
            "Grouped permutation importance measures model dependence. "
            "Coefficients show learned relationships in standardized "
            "feature space. Neither establishes causal effects."
        )

        # Grouped permutation comparison
        st.markdown(
            "### Grouped permutation importance"
        )

        comparison = read_csv(
            ridge_dir / "model_group_importance_comparison.csv"
        )

        if comparison is not None:
            fig = go.Figure()

            fig.add_bar(
                x=comparison["name"],
                y=comparison["Ridge"],
                name="Ridge"
            )

            fig.add_bar(
                x=comparison["name"],
                y=comparison["LightGBM"],
                name="LightGBM"
            )

            fig.update_layout(
                barmode="group",
                title="Grouped permutation importance: Ridge vs LightGBM",
                yaxis_title="RMSE increase (MW)",
                xaxis_title="Feature group",
                height=450,
                legend=dict(orientation="h")
            )

            show(fig)

            st.caption(
                "Larger RMSE increases indicate greater model dependence "
                "on that feature group. Feature groups overlap, so their "
                "values must not be added together."
            )

            display_comparison = comparison[
                ["name", "Ridge", "LightGBM"]
            ].copy()

            display_comparison = display_comparison.rename(
                columns={"name": "Feature group"}
            )

            table(
                display_comparison.style.format({
                    "Ridge": "{:,.2f}",
                    "LightGBM": "{:,.2f}"
                })
            )

        else:
            st.warning(
                "Ridge-LightGBM comparison file not found. "
                "Run scripts/04_explain_ridge.py."
            )

        # Ridge coefficients
        st.markdown(
            "### Ridge standardized coefficients"
        )

        coefficients = read_csv(
            ridge_dir / "ridge_coefficients.csv"
        )

        if coefficients is not None:
            top = coefficients.nlargest(
                20,
                "abs_coefficient"
            ).sort_values(
                "coefficient_mw_per_1sd"
            )

            fig = go.Figure(
                go.Bar(
                    x=top["coefficient_mw_per_1sd"],
                    y=top["feature"],
                    orientation="h"
                )
            )

            fig.update_layout(
                title="Top 20 Ridge coefficients",
                xaxis_title="Coefficient (MW per 1 SD feature change)",
                yaxis_title="Feature",
                height=650
            )

            show(fig)

            st.caption(
                "Positive coefficients indicate a positive learned "
                "association with predicted demand; negative coefficients "
                "indicate a negative association, conditional on the other "
                "model features. Magnitudes are expressed per one standard "
                "deviation of the feature. Correlated demand lags can produce "
                "large coefficients with opposing signs, so interpret "
                "individual coefficients cautiously."
            )

            display_coefficients = coefficients.nlargest(
                20,
                "abs_coefficient"
            )[
                [
                    "feature",
                    "coefficient_mw_per_1sd",
                    "abs_coefficient"
                ]
            ].copy()

            display_coefficients = display_coefficients.rename(
                columns={
                    "feature": "Feature",
                    "coefficient_mw_per_1sd": "Coefficient (MW / 1 SD)",
                    "abs_coefficient": "Absolute coefficient"
                }
            )

            table(
                display_coefficients.style.format({
                    "Coefficient (MW / 1 SD)": "{:,.2f}",
                    "Absolute coefficient": "{:,.2f}"
                })
            )

        else:
            st.warning(
                "Ridge coefficient file not found. "
                "Run scripts/04_explain_ridge.py."
            )


# ------------------------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "24-hour forecast": page_forecast,
    "Scenario simulator": page_simulator,
    "Model insights": page_insights
}

page = st.sidebar.radio(
    "GridShift",
    list(PAGES)
)

if page != "Overview" and not (
    C.CLEAN_PARQUET.exists() or C.PROCESSED_CSV.exists()
):
    st.error(
        f"Data not found at {C.PROCESSED_CSV}. "
        "Edit gridshift/config.py or set GRIDSHIFT_DATA."
    )
    st.stop()

PAGES[page]()
