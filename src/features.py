"""
features.py
Spatial feature engineering for the Hillsborough County housing predictor.
Merges census tracts with ACS, Zillow, OSM, and FEMA data, then builds
spatial lag and proximity features for model training.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point


# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXTERNAL = ROOT / "data" / "external"


def load_all_data():
    """Load all processed datasets from disk."""
    print("Loading processed datasets...")

    tracts = gpd.read_file(PROCESSED / "hillsborough_tracts.geojson")
    print("  Tracts loaded: {}".format(len(tracts)))

    acs = pd.read_csv(PROCESSED / "acs_hillsborough.csv")
    print("  ACS loaded: {} rows".format(len(acs)))

    zillow = pd.read_csv(PROCESSED / "zillow_hillsborough.csv")
    print("  Zillow loaded: {} rows".format(len(zillow)))

    osm = {}
    for name in ["schools", "parks", "transit", "grocery", "hospitals"]:
        path = PROCESSED / "osm_{}.geojson".format(name)
        if path.exists():
            osm[name] = gpd.read_file(path)
            print("  OSM {} loaded: {} features".format(name, len(osm[name])))
        else:
            print("  OSM {} not found, skipping.".format(name))

    fema_path = PROCESSED / "fema_flood_hillsborough.geojson"
    if fema_path.exists():
        fema = gpd.read_file(fema_path)
        print("  FEMA loaded: {} features".format(len(fema)))
    else:
        fema = gpd.GeoDataFrame()
        print("  FEMA not found, skipping.")

    return tracts, acs, zillow, osm, fema


def merge_acs_to_tracts(tracts, acs):
    """Merge ACS demographic data onto census tract geometries."""
    print("Merging ACS data to tracts...")

    # Ensure GEOID formats match
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    acs["GEOID"] = acs["GEOID"].astype(str).str.zfill(11)

    merged = tracts.merge(acs, on="GEOID", how="left")
    print("  Merged shape: {}".format(merged.shape))
    return merged


def add_zillow_target(gdf, zillow):
    """
    Add Zillow ZHVI as the target variable.
    Since Zillow county-level data is one row, we use ACS median_home_value
    as the primary target and ZHVI as a county-level benchmark column.
    """
    print("Adding Zillow ZHVI benchmark...")

    if len(zillow) > 0 and "zhvi_latest" in zillow.columns:
        zhvi_val = zillow["zhvi_latest"].values[0]
        zhvi_date = zillow["zhvi_date"].values[0] if "zhvi_date" in zillow.columns else "unknown"
        gdf["zhvi_county_benchmark"] = zhvi_val
        print("  ZHVI county benchmark: ${:,.0f} (as of {})".format(zhvi_val, zhvi_date))
    else:
        gdf["zhvi_county_benchmark"] = np.nan
        print("  ZHVI not available, set to NaN.")

    # Use ACS median_home_value as the tract-level target
    gdf["target_home_value"] = pd.to_numeric(gdf["median_home_value"], errors="coerce")

    # Drop tracts with no target
    before = len(gdf)
    gdf = gdf[gdf["target_home_value"] > 0].copy()
    after = len(gdf)
    print("  Dropped {} tracts with missing target. {} remaining.".format(before - after, after))

    return gdf


def compute_proximity_features(gdf, osm):
    """
    For each census tract centroid, compute distance to nearest OSM amenity
    and count of amenities within a radius.
    """
    print("Computing proximity features...")

    # Project to a meter-based CRS for accurate distance calculation
    gdf_proj = gdf.to_crs(epsg=3857).copy()
    gdf_proj["centroid"] = gdf_proj.geometry.centroid

    radius_m = 2000  # 2 km radius

    for name, amenity_gdf in osm.items():
        if amenity_gdf is None or len(amenity_gdf) == 0:
            gdf["dist_to_{}".format(name)] = np.nan
            gdf["count_{}_2km".format(name)] = 0
            continue

        amenity_proj = amenity_gdf.to_crs(epsg=3857).copy()

        # Use centroid for polygon amenities
        amenity_proj["geometry"] = amenity_proj.geometry.apply(
            lambda g: g.centroid if g.geom_type != "Point" else g
        )

        dist_list = []
        count_list = []

        for centroid in gdf_proj["centroid"]:
            distances = amenity_proj.geometry.distance(centroid)
            dist_list.append(distances.min())
            count_list.append((distances <= radius_m).sum())

        gdf["dist_to_{}".format(name)] = dist_list
        gdf["count_{}_2km".format(name)] = count_list
        print("  Proximity features added for: {}".format(name))

    return gdf


def add_flood_zone_feature(gdf, fema):
    """Add percentage of tract area in FEMA high-risk flood zone (Zone A or AE)."""
    print("Adding FEMA flood zone features...")

    if fema is None or len(fema) == 0:
        gdf["pct_flood_zone"] = 0.0
        gdf["in_flood_zone"] = 0
        print("  No FEMA data available, set to 0.")
        return gdf

    gdf_proj = gdf.to_crs(epsg=3857).copy()

    high_risk = fema[fema["FLD_ZONE"].isin(["A", "AE", "AH", "AO", "VE", "V"])].copy()
    if len(high_risk) == 0:
        gdf["pct_flood_zone"] = 0.0
        gdf["in_flood_zone"] = 0
        return gdf

    high_risk_proj = high_risk.to_crs(epsg=3857).copy()
    flood_union = high_risk_proj.geometry.unary_union

    pct_list = []
    for geom in gdf_proj.geometry:
        try:
            intersection = geom.intersection(flood_union)
            pct = intersection.area / geom.area if geom.area > 0 else 0.0
        except Exception:
            pct = 0.0
        pct_list.append(pct)

    gdf["pct_flood_zone"] = pct_list
    gdf["in_flood_zone"] = (gdf["pct_flood_zone"] > 0.1).astype(int)
    print("  Flood zone features added. {} tracts partially in flood zone.".format(
        gdf["in_flood_zone"].sum()
    ))

    return gdf


def add_spatial_lag_features(gdf):
    """
    Add spatial lag features using queen contiguity weights.
    These capture neighborhood effects — a key spatial ML feature.
    """
    print("Adding spatial lag features...")

    try:
        from libpysal.weights import Queen
        import esda

        gdf_proj = gdf.to_crs(epsg=3857).copy()
        w = Queen.from_dataframe(gdf_proj, silence_warnings=True)
        w.transform = "r"  # Row-standardize

        lag_cols = [
            "median_household_income",
            "total_population",
            "pct_bachelors",
            "unemployment_rate",
            "pct_owner_occupied",
            "median_age",
        ]

        for col in lag_cols:
            if col in gdf.columns:
                vals = gdf[col].fillna(gdf[col].median()).values
                lag_vals = [
                    np.sum([w.weights[i][j] * vals[w.neighbors[i][k]]
                            for k, j in enumerate(w.neighbors[i])])
                    if w.neighbors[i] else vals[i]
                    for i in range(len(vals))
                ]
                gdf["lag_{}".format(col)] = lag_vals
                print("  Spatial lag added: lag_{}".format(col))

        # Moran's I for target variable
        target_vals = gdf["target_home_value"].fillna(gdf["target_home_value"].median()).values
        mi = esda.Moran(target_vals, w)
        print("  Moran's I for home values: {:.4f} (p={:.4f})".format(mi.I, mi.p_sim))
        gdf["morans_i"] = mi.I

    except Exception as e:
        print("  Spatial lag skipped: {}".format(e))

    return gdf


def add_tract_shape_features(gdf):
    """Add geometric features about each tract."""
    print("Adding tract shape features...")

    gdf_proj = gdf.to_crs(epsg=3857).copy()
    gdf["tract_area_sqkm"] = gdf_proj.geometry.area / 1e6
    gdf["tract_perimeter_km"] = gdf_proj.geometry.length / 1000
    gdf["compactness"] = (
        4 * np.pi * gdf_proj.geometry.area /
        (gdf_proj.geometry.length ** 2)
    ).clip(0, 1)

    # Distance to Tampa CBD (approximate centroid)
    tampa_cbd = Point(-82.4572, 27.9506)
    tampa_cbd_proj = gpd.GeoSeries([tampa_cbd], crs=4326).to_crs(epsg=3857).iloc[0]
    gdf["dist_to_cbd_km"] = gdf_proj.geometry.centroid.distance(tampa_cbd_proj) / 1000
    print("  Tract shape and CBD distance features added.")

    return gdf


def build_feature_matrix(gdf):
    """Select and clean final feature columns for modeling."""
    print("Building final feature matrix...")

    feature_cols = [
        # ACS demographics
        "total_population",
        "median_household_income",
        "median_age",
        "median_gross_rent",
        "pct_bachelors",
        "unemployment_rate",
        "pct_owner_occupied",
        "pct_new_housing",
        "pct_long_commute",
        # Proximity
        "dist_to_schools",
        "dist_to_parks",
        "dist_to_transit",
        "dist_to_grocery",
        "dist_to_hospitals",
        "count_schools_2km",
        "count_parks_2km",
        "count_transit_2km",
        "count_grocery_2km",
        "count_hospitals_2km",
        # Flood
        "pct_flood_zone",
        "in_flood_zone",
        # Spatial lag
        "lag_median_household_income",
        "lag_total_population",
        "lag_pct_bachelors",
        "lag_unemployment_rate",
        "lag_pct_owner_occupied",
        "lag_median_age",
        # Geometry
        "tract_area_sqkm",
        "compactness",
        "dist_to_cbd_km",
        # Zillow benchmark
        "zhvi_county_benchmark",
    ]

    # Only keep columns that exist
    available = [c for c in feature_cols if c in gdf.columns]
    missing = [c for c in feature_cols if c not in gdf.columns]
    if missing:
        print("  Warning: missing columns (will be skipped): {}".format(missing))

    target_col = "target_home_value"
    keep_cols = ["GEOID", "geometry", target_col] + available

    result = gdf[keep_cols].copy()

    # Fill remaining NaN with column medians
    for col in available:
        if result[col].isna().any():
            median_val = result[col].median()
            result[col] = result[col].fillna(median_val)

    print("  Feature matrix shape: {}".format(result.shape))
    print("  Features: {}".format(len(available)))
    print("  Target: {}".format(target_col))

    return result, available


def run_feature_engineering():
    """Full feature engineering pipeline."""
    print("=" * 60)
    print("HILLSBOROUGH HOUSING PREDICTOR -- Feature Engineering")
    print("=" * 60)

    # Load
    tracts, acs, zillow, osm, fema = load_all_data()

    # Merge
    gdf = merge_acs_to_tracts(tracts, acs)
    gdf = add_zillow_target(gdf, zillow)

    # Spatial features
    gdf = compute_proximity_features(gdf, osm)
    gdf = add_flood_zone_feature(gdf, fema)
    gdf = add_tract_shape_features(gdf)
    gdf = add_spatial_lag_features(gdf)

    # Final matrix
    feature_gdf, feature_cols = build_feature_matrix(gdf)

    # Save
    out_path = PROCESSED / "features.geojson"
    feature_gdf.to_file(out_path, driver="GeoJSON")
    print("\nFeature GeoDataFrame saved -> {}".format(out_path))

    # Also save as CSV for model training
    csv_path = PROCESSED / "features.csv"
    feature_gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    print("Feature CSV saved -> {}".format(csv_path))

    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("  Total tracts: {}".format(len(feature_gdf)))
    print("  Total features: {}".format(len(feature_cols)))
    print("=" * 60)

    return feature_gdf, feature_cols


if __name__ == "__main__":
    run_feature_engineering()
