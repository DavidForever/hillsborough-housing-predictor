"""
streamlit_app.py
Interactive Streamlit dashboard for the Hillsborough County Housing Price Predictor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import joblib
import json
import matplotlib.pyplot as plt

from pathlib import Path
from streamlit_folium import st_folium

# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PLOTS = ROOT / "outputs" / "plots"

st.set_page_config(
    page_title="Hillsborough Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header { font-size: 2rem; font-weight: 700; color: #1f4e79; }
.sub-header { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_predictions():
    path = PROCESSED / "predictions.geojson"
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=4326)
    return gdf


@st.cache_resource
def load_metrics():
    try:
        return pd.read_csv(MODELS / "metrics.csv")
    except Exception:
        return None


def format_currency(val):
    if pd.isna(val):
        return "N/A"
    return "${:,.0f}".format(val)


def build_map(gdf, value_col, title):
    """Build Folium choropleth map with no lambda functions."""

    # Reproject to get accurate centroids
    gdf_proj = gdf.to_crs(epsg=3857)
    center_lat = float(gdf_proj.geometry.centroid.to_crs(epsg=4326).y.mean())
    center_lon = float(gdf_proj.geometry.centroid.to_crs(epsg=4326).x.mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="CartoDB positron"
    )

    # Prepare data
    map_data = gdf[["GEOID", value_col]].dropna().copy()
    map_data["GEOID"] = map_data["GEOID"].astype(str)

    # Build geojson string
    geo_str = gdf.to_json()

    # Choropleth layer — no lambdas
    choropleth = folium.Choropleth(
        geo_data=geo_str,
        data=map_data,
        columns=["GEOID", value_col],
        key_on="feature.properties.GEOID",
        fill_color="RdYlGn",
        fill_opacity=0.75,
        line_opacity=0.4,
        line_weight=0.5,
        legend_name=title,
        nan_fill_color="lightgrey",
        nan_fill_opacity=0.4,
        highlight=True,
    )
    choropleth.add_to(m)

    # Add tooltip data directly to the choropleth geojson
    tooltip_fields = []
    tooltip_aliases = []

    field_alias_map = {
        "GEOID": "Tract ID",
        "target_home_value": "Actual Value ($)",
        "predicted_home_value": "Predicted Value ($)",
        "pct_error": "Error (%)",
        "top_shap_feature": "Top SHAP Driver",
        "median_household_income": "Med. Income ($)",
        "total_population": "Population",
    }

    for field, alias in field_alias_map.items():
        if field in gdf.columns:
            tooltip_fields.append(field)
            tooltip_aliases.append(alias)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
        )
    )

    return m


def main():
    st.markdown('<div class="main-header">🏠 Hillsborough County Housing Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Geospatial ML model predicting median home values by census tract — Tampa, FL</div>', unsafe_allow_html=True)

    gdf = load_predictions()
    metrics = load_metrics()

    if gdf is None:
        st.error("No predictions found. Please run train_and_plot.py first.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### Map Controls")

        map_mode = st.selectbox(
            "Display Variable",
            options=[
                "Predicted Home Value",
                "Actual Home Value",
                "Prediction Error (%)",
                "Median Household Income",
                "% Bachelor Degrees",
                "Unemployment Rate",
                "% Owner Occupied",
                "Distance to CBD (km)",
                "% in Flood Zone",
            ]
        )

        st.markdown("---")
        st.markdown("### Filter Tracts")
        min_val = int(gdf["target_home_value"].min())
        max_val = int(gdf["target_home_value"].max())
        value_range = st.slider(
            "Actual Home Value Range ($)",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=10000,
            format="$%d"
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Data Sources:**
        - ACS 2022 5-Year Estimates
        - TIGER/Line 2023 Shapefiles
        - Zillow ZHVI
        - OpenStreetMap
        - FEMA NFHL

        **Model:** XGBoost + SHAP

        **Built by:** David Favors
        **GitHub:** [hillsborough-housing-predictor](https://github.com/DavidForever/hillsborough-housing-predictor)
        """)

    # Filter
    filtered = gdf[
        (gdf["target_home_value"] >= value_range[0]) &
        (gdf["target_home_value"] <= value_range[1])
    ].copy()

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Census Tracts", "{:,}".format(len(filtered)))
    with c2:
        st.metric("Median Predicted", format_currency(filtered["predicted_home_value"].median()))
    with c3:
        st.metric("Model R2", "{:.4f}".format(metrics["r2"].values[0]) if metrics is not None else "N/A")
    with c4:
        st.metric("MAE", format_currency(metrics["mae"].values[0]) if metrics is not None else "N/A")
    with c5:
        st.metric("MAPE", "{:.2f}%".format(metrics["mape"].values[0]) if metrics is not None else "N/A")

    st.markdown("---")

    col_map = {
        "Predicted Home Value":    ("predicted_home_value",    "Predicted Home Value ($)"),
        "Actual Home Value":       ("target_home_value",       "Actual Home Value ($)"),
        "Prediction Error (%)":    ("pct_error",               "Prediction Error (%)"),
        "Median Household Income": ("median_household_income", "Median Household Income ($)"),
        "% Bachelor Degrees":      ("pct_bachelors",           "% Bachelor Degrees"),
        "Unemployment Rate":       ("unemployment_rate",       "Unemployment Rate"),
        "% Owner Occupied":        ("pct_owner_occupied",      "% Owner Occupied"),
        "Distance to CBD (km)":    ("dist_to_cbd_km",          "Distance to Tampa CBD (km)"),
        "% in Flood Zone":         ("pct_flood_zone",          "% Tract in Flood Zone"),
    }

    value_col, map_title = col_map[map_mode]

    map_col, analysis_col = st.columns([3, 1])

    with map_col:
        st.markdown("#### {} by Census Tract".format(map_title))
        if value_col in filtered.columns:
            folium_map = build_map(filtered, value_col, map_title)
            st_folium(folium_map, width=800, height=550)
        else:
            st.warning("Column '{}' not available.".format(value_col))

    with analysis_col:
        st.markdown("#### Distribution")
        if value_col in filtered.columns:
            fig, ax = plt.subplots(figsize=(4, 3))
            filtered[value_col].dropna().hist(bins=20, ax=ax, color="steelblue", edgecolor="white")
            ax.set_xlabel(map_title, fontsize=8)
            ax.set_ylabel("Tracts", fontsize=8)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("#### Summary Stats")
        col = filtered[value_col].dropna()
        st.dataframe(
            pd.DataFrame({
                "Stat": ["Min", "25%", "Median", "75%", "Max"],
                "Value": [
                    format_currency(col.min()),
                    format_currency(col.quantile(0.25)),
                    format_currency(col.median()),
                    format_currency(col.quantile(0.75)),
                    format_currency(col.max()),
                ]
            }),
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")

    # SHAP plots
    st.markdown("### Model Explainability (SHAP)")
    s1, s2 = st.columns(2)
    with s1:
        p = PLOTS / "shap_summary.png"
        if p.exists():
            st.image(str(p), caption="SHAP Summary Plot", use_column_width=True)
    with s2:
        p = PLOTS / "shap_bar.png"
        if p.exists():
            st.image(str(p), caption="SHAP Feature Importance", use_column_width=True)

    st.markdown("---")

    # Predictions table
    st.markdown("### Tract-Level Predictions")
    display_cols = [
        "GEOID", "target_home_value", "predicted_home_value",
        "residual", "pct_error", "top_shap_feature",
        "median_household_income", "total_population"
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[display_cols].copy()

    rename_map = {
        "GEOID": "Tract ID",
        "target_home_value": "Actual ($)",
        "predicted_home_value": "Predicted ($)",
        "residual": "Residual ($)",
        "pct_error": "Error (%)",
        "top_shap_feature": "Top Driver",
        "median_household_income": "Med. Income ($)",
        "total_population": "Population",
    }
    display_df = display_df.rename(columns=rename_map)

    for col in ["Actual ($)", "Predicted ($)", "Residual ($)", "Med. Income ($)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: "${:,.0f}".format(x) if not pd.isna(x) else "N/A"
            )

    if "Error (%)" in display_df.columns:
        st.dataframe(
            display_df.sort_values("Error (%)", key=abs, ascending=False),
            use_container_width=True,
            height=350
        )
    else:
        st.dataframe(display_df, use_container_width=True, height=350)

    csv = filtered.drop(columns=["geometry"]).to_csv(index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="hillsborough_predictions.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Performance plots
    st.markdown("### Model Performance")
    p1, p2 = st.columns(2)
    with p1:
        p = PLOTS / "predictions_vs_actual.png"
        if p.exists():
            st.image(str(p), caption="Predicted vs Actual", use_column_width=True)
    with p2:
        p = PLOTS / "residuals.png"
        if p.exists():
            st.image(str(p), caption="Residual Analysis", use_column_width=True)


if __name__ == "__main__":
    main()
