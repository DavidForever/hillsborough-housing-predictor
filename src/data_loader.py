"""
data_loader.py
Fetches and saves all raw data for the Hillsborough County housing predictor:
  - Census tract geometries (TIGER/Line 2023)
  - ACS 2022 5-year demographic estimates
  - Zillow ZHVI median home values
  - OpenStreetMap amenities (schools, parks, transit, grocery, hospitals)
  - FEMA flood zones
"""

import os
import requests
import zipfile
import geopandas as gpd
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

# Paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"

for p in [RAW, EXTERNAL, PROCESSED]:
    p.mkdir(parents=True, exist_ok=True)

# Constants
STATE_FIPS = "12"
COUNTY_FIPS = "057"
FULL_FIPS = STATE_FIPS + COUNTY_FIPS


def download_tiger_tracts():
    """Download Florida census tracts from TIGER/Line and filter to Hillsborough."""
    url = (
        "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/"
        "tl_2023_12_tract.zip"
    )
    zip_path = RAW / "tl_2023_12_tract.zip"

    if not zip_path.exists():
        print("Downloading TIGER/Line tract shapefile...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        print("  Saved ->", zip_path)

    extract_dir = RAW / "tiger_tracts"
    if not extract_dir.exists():
        print("Extracting shapefile...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

    shp = next(extract_dir.glob("*.shp"))
    gdf = gpd.read_file(shp)
    gdf = gdf[gdf["COUNTYFP"] == COUNTY_FIPS].copy()
    gdf = gdf.to_crs(epsg=4326)

    out = PROCESSED / "hillsborough_tracts.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print("  Census tracts saved ->", out, " ({} tracts)".format(len(gdf)))
    return gdf


def download_acs_data():
    """Fetch ACS demographic variables for all Hillsborough tracts."""
    variables = {
        "B01003_001E": "total_population",
        "B19013_001E": "median_household_income",
        "B25077_001E": "median_home_value",
        "B15003_022E": "bachelors_degree",
        "B15003_001E": "education_total",
        "B23025_005E": "unemployed",
        "B23025_001E": "labor_force_total",
        "B25003_002E": "owner_occupied",
        "B25003_001E": "housing_total",
        "B01002_001E": "median_age",
        "B25064_001E": "median_gross_rent",
        "B08303_001E": "commute_total",
        "B08303_013E": "commute_60plus_min",
        "B25034_001E": "housing_units_total",
        "B25034_002E": "built_2020_later",
        "B25034_003E": "built_2010_2019",
    }

    var_str = ",".join(variables.keys())
    url = (
        "https://api.census.gov/data/2022/acs/acs5"
        "?get=NAME,{}"
        "&for=tract:*"
        "&in=state:{}%20county:{}"
        "&key={}"
    ).format(var_str, STATE_FIPS, COUNTY_FIPS, CENSUS_API_KEY)

    print("Fetching ACS 2022 data from Census API...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns=variables)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    numeric_cols = list(variables.values())
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df["pct_bachelors"] = df["bachelors_degree"] / df["education_total"].replace(0, pd.NA)
    df["unemployment_rate"] = df["unemployed"] / df["labor_force_total"].replace(0, pd.NA)
    df["pct_owner_occupied"] = df["owner_occupied"] / df["housing_total"].replace(0, pd.NA)
    df["pct_new_housing"] = (df["built_2020_later"] + df["built_2010_2019"]) / df["housing_units_total"].replace(0, pd.NA)
    df["pct_long_commute"] = df["commute_60plus_min"] / df["commute_total"].replace(0, pd.NA)

    out = PROCESSED / "acs_hillsborough.csv"
    df.to_csv(out, index=False)
    print("  ACS data saved ->", out, " ({} tracts)".format(len(df)))
    return df


def download_zillow_data():
    """Download Zillow ZHVI county-level median home values."""
    county_url = (
        "https://files.zillowstatic.com/research/public_csvs/zhvi/"
        "County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    )

    out_path = EXTERNAL / "zillow_county_zhvi.csv"
    if not out_path.exists():
        print("Downloading Zillow ZHVI county data...")
        r = requests.get(county_url, timeout=120)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        print("  Saved ->", out_path)

    df = pd.read_csv(out_path)
    hill = df[(df["StateName"] == "Florida") & (df["RegionName"] == "Hillsborough County")].copy()

    date_cols = [c for c in df.columns if c.startswith("20")]
    latest_col = date_cols[-1]
    if len(hill) > 0:
        hill["zhvi_latest"] = hill[latest_col].values[0]
    hill["zhvi_date"] = latest_col

    out = PROCESSED / "zillow_hillsborough.csv"
    hill.to_csv(out, index=False)
    print("  Zillow data saved ->", out, " (latest: {})".format(latest_col))
    return hill


def download_osm_amenities():
    """Fetch POIs from OSM for Hillsborough County."""
    try:
        import osmnx as ox
    except ImportError:
        print("osmnx not installed -- skipping OSM download.")
        return {}

    place = "Hillsborough County, Florida, USA"
    tags_map = {
        "schools": {"amenity": "school"},
        "parks": {"leisure": "park"},
        "transit": {"public_transport": "stop_position"},
        "grocery": {"shop": "supermarket"},
        "hospitals": {"amenity": "hospital"},
    }

    results = {}
    for name, tags in tags_map.items():
        out_path = PROCESSED / "osm_{}.geojson".format(name)
        if out_path.exists():
            print("  OSM {} already downloaded, loading from cache.".format(name))
            results[name] = gpd.read_file(out_path)
            continue
        try:
            print("  Fetching OSM: {}...".format(name))
            gdf = ox.features_from_place(place, tags=tags)
            gdf = gdf[["geometry"]].copy()
            gdf = gdf.to_crs(epsg=4326)
            gdf.to_file(out_path, driver="GeoJSON")
            results[name] = gdf
            print("    Saved {} {} -> {}".format(len(gdf), name, out_path))
        except Exception as e:
            print("    Warning: could not fetch {}: {}".format(name, e))

    return results


def download_fema_flood():
    """Download FEMA NFHL flood zones for Hillsborough County."""
    out_path = PROCESSED / "fema_flood_hillsborough.geojson"
    if out_path.exists():
        print("  FEMA flood zones already downloaded, loading from cache.")
        return gpd.read_file(out_path)

    print("Fetching FEMA flood zones...")
    url = (
        "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
        "?where=DFIRM_ID+LIKE+'12057%25'"
        "&outFields=FLD_ZONE,ZONE_SUBTY"
        "&returnGeometry=true"
        "&f=geojson"
        "&resultRecordCount=2000"
    )
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        out_path.write_text(r.text, encoding="utf-8")
        gdf = gpd.read_file(out_path)
        print("  FEMA flood zones saved ->", out_path, " ({} features)".format(len(gdf)))
        return gdf
    except Exception as e:
        print("  Warning: FEMA download failed:", e)
        return gpd.GeoDataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("HILLSBOROUGH HOUSING PREDICTOR -- Data Loader")
    print("=" * 60)

    print("\n[1/5] TIGER/Line Census Tracts")
    tracts = download_tiger_tracts()

    print("\n[2/5] ACS 2022 Demographic Data")
    acs = download_acs_data()

    print("\n[3/5] Zillow ZHVI Home Values")
    zillow = download_zillow_data()

    print("\n[4/5] OpenStreetMap Amenities")
    osm = download_osm_amenities()

    print("\n[5/5] FEMA Flood Zones")
    fema = download_fema_flood()

    print("\n" + "=" * 60)
    print("All data downloaded successfully!")
    print("  Tracts: {}".format(len(tracts)))
    print("  ACS:    {} tracts".format(len(acs)))
    print("  OSM:    {}".format(", ".join("{}={}".format(k, len(v)) for k, v in osm.items())))
    print("=" * 60)
