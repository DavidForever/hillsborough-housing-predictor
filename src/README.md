# 🏠 Hillsborough County Housing Price Predictor

A geospatial machine learning project that predicts median home values by census tract in Hillsborough County, Florida using spatial features, demographic data, and amenity proximity.

Built as a portfolio project demonstrating skills in GIS, spatial data engineering, machine learning, and interactive visualization.

---

## 📸 Dashboard Preview

> Run the Streamlit app to see the interactive map dashboard with choropleth maps, SHAP explainability plots, and tract-level predictions.

---

## 🎯 Project Overview

This project combines multiple public data sources to engineer spatially-aware features and train an XGBoost regression model to predict median home values at the census tract level. The model is explainable via SHAP values and served through an interactive Streamlit dashboard with Folium maps.

**Key question:** What spatial and demographic factors best explain variation in home values across Hillsborough County census tracts?

---

## 🗂️ Project Structure

```
hillsborough-housing-predictor/
├── .env                        # Census API key (not committed)
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                    # Downloaded shapefiles and zips
│   ├── processed/              # Cleaned GeoDataFrames and CSVs
│   └── external/               # Zillow ZHVI county data
│
├── src/
│   ├── data_loader.py          # Downloads all raw data
│   ├── features.py             # Spatial feature engineering
│   ├── model.py                # XGBoost training and SHAP
│   └── streamlit_app.py        # Interactive dashboard
│
├── models/                     # Saved model artifacts
└── outputs/
    ├── maps/                   # Folium HTML maps
    └── plots/                  # SHAP and evaluation plots
```

---

## 📦 Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| [Census TIGER/Line 2023](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | Census tract geometries for Florida | Free download |
| [ACS 2022 5-Year Estimates](https://www.census.gov/data/developers/data-sets/acs-5year.html) | Demographic and housing variables by tract | Free API key |
| [Zillow ZHVI](https://www.zillow.com/research/data/) | Median home values at county level | Free download |
| [OpenStreetMap via OSMnx](https://osmnx.readthedocs.io/) | Schools, parks, transit, grocery, hospitals | Free |
| [FEMA NFHL](https://msc.fema.gov/portal/home) | Flood hazard zones for Hillsborough County | Free API |

---

## 🧠 Features Engineered

### Demographic (ACS)
- Median household income
- Median age and gross rent
- Percent with bachelor's degree
- Unemployment rate
- Percent owner-occupied housing
- Percent new housing stock (built 2010+)
- Percent long commuters (60+ min)

### Proximity (OpenStreetMap)
- Distance to nearest school, park, transit stop, grocery store, hospital
- Count of each amenity within 2km radius

### Spatial
- Distance to Tampa CBD
- Tract area and compactness ratio
- Spatial lag features (Queen contiguity weights)
- Moran's I spatial autocorrelation

### Risk
- Percent of tract area in FEMA high-risk flood zone
- Binary flood zone indicator

---

## 🤖 Model

| Item | Detail |
|------|--------|
| **Algorithm** | XGBoost Regressor |
| **Baseline** | Ridge Regression |
| **Validation** | 5-Fold Cross Validation |
| **Explainability** | SHAP TreeExplainer |
| **Target** | ACS Median Home Value by Census Tract |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/DavidForever/hillsborough-housing-predictor.git
cd hillsborough-housing-predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install streamlit-folium
```

### 4. Set up your Census API key

Get a free key at: https://api.census.gov/data/key_signup.html

Create a `.env` file in the project root:

```
CENSUS_API_KEY=your_key_here
```

### 5. Run the pipeline in order

```bash
# Step 1 - Download all data
python src/data_loader.py

# Step 2 - Engineer spatial features
python src/features.py

# Step 3 - Train the model
python src/model.py

# Step 4 - Launch the dashboard
streamlit run src/streamlit_app.py
```

---

## 📊 Results

After running the full pipeline, open the Streamlit dashboard to explore:

- Choropleth map of predicted vs actual home values
- SHAP summary plot showing top predictive features
- Tract-level predictions table with error metrics
- Residual analysis plots
- Downloadable predictions CSV

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Geospatial** | GeoPandas, OSMnx, Folium, libpysal, ESDA, PyProj, Shapely |
| **Machine Learning** | XGBoost, scikit-learn, SHAP |
| **Data** | Pandas, NumPy, Requests, python-dotenv |
| **Visualization** | Matplotlib, Seaborn, Plotly, Folium |
| **Dashboard** | Streamlit, streamlit-folium |
| **Environment** | Python 3.10, pip, venv |

---

## 👤 Author

**David**
- GitHub: [@DavidForever](https://github.com/DavidForever)
- Project built as a GIS-to-Data Science portfolio transition piece

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- U.S. Census Bureau for ACS and TIGER/Line data
- Zillow Research for ZHVI data
- OpenStreetMap contributors
- FEMA for National Flood Hazard Layer data
- The GeoPandas, OSMnx, and libpysal open source communities
