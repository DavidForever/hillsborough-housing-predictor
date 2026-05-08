import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PLOTS = ROOT / "outputs" / "plots"

print("Loading artifacts...")
df = pd.read_csv(PROCESSED / "features.csv")
gdf = gpd.read_file(PROCESSED / "features.geojson")
feature_cols = joblib.load(MODELS / "feature_cols.pkl")
xgb_model = joblib.load(MODELS / "xgb_model.pkl")
imputer = joblib.load(MODELS / "imputer.pkl")

X_raw = df[feature_cols].values
X_clean = imputer.transform(X_raw)
y = df["target_home_value"]
actual_feature_cols = feature_cols[:X_clean.shape[1]]
y_pred = xgb_model.predict(X_clean)

print("Plotting feature importance...")
importance = pd.Series(xgb_model.feature_importances_, index=actual_feature_cols).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 7))
importance.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 Feature Importances (XGBoost)", fontsize=14)
plt.tight_layout()
fig.savefig(PLOTS / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Feature importance saved.")

print("Computing SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_clean)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_clean, feature_names=actual_feature_cols, show=False)
plt.tight_layout()
plt.savefig(PLOTS / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  SHAP summary saved.")

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_clean, feature_names=actual_feature_cols, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(PLOTS / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  SHAP bar saved.")

print("Plotting predictions vs actual...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y, y_pred, alpha=0.6, color="steelblue", edgecolors="none")
min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
ax.set_xlabel("Actual ($)")
ax.set_ylabel("Predicted ($)")
ax.set_title("Predicted vs Actual Home Values")
r2 = r2_score(y, y_pred)
ax.text(0.05, 0.92, "R2 = {:.4f}".format(r2), transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.tight_layout()
fig.savefig(PLOTS / "predictions_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Predictions vs actual saved.")

print("Plotting residuals...")
residuals = y - y_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred, residuals, alpha=0.5, color="steelblue", edgecolors="none")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Residuals vs Predicted")
axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Residual Distribution")
plt.tight_layout()
fig.savefig(PLOTS / "residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Residuals saved.")

print("Saving predictions GeoDataFrame...")
gdf["predicted_home_value"] = y_pred
gdf["residual"] = gdf["target_home_value"] - gdf["predicted_home_value"]
gdf["pct_error"] = (gdf["residual"] / gdf["target_home_value"]) * 100
shap_df = pd.DataFrame(shap_values, columns=actual_feature_cols)
gdf["top_shap_feature"] = shap_df.abs().idxmax(axis=1)
gdf["top_shap_value"] = shap_df.abs().max(axis=1)
gdf.to_file(PROCESSED / "predictions.geojson", driver="GeoJSON")
print("  Predictions GeoDataFrame saved.")

print("\nAll done! Ready to launch Streamlit.")
