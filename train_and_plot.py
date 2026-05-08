"""
train_and_plot.py
Complete training pipeline - run this instead of model.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PLOTS = ROOT / "outputs" / "plots"

for p in [MODELS, PLOTS]:
    p.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading feature matrix...")
df = pd.read_csv(PROCESSED / "features.csv")
gdf = gpd.read_file(PROCESSED / "features.geojson")
print("  Loaded {} tracts".format(len(df)))

# ── Features ──────────────────────────────────────────────────────────────
exclude = ["GEOID", "target_home_value", "geometry"]
feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
print("  Feature columns: {}".format(len(feature_cols)))

X_raw = df[feature_cols].values
y = df["target_home_value"].values

# ── Impute ────────────────────────────────────────────────────────────────
print("Imputing NaN values...")
imputer = SimpleImputer(strategy="median")
X_clean = imputer.fit_transform(X_raw)
print("  X shape after imputation: {}".format(X_clean.shape))

# Sync feature_cols to actual number of columns after imputation
feature_cols = feature_cols[:X_clean.shape[1]]

print("\nTarget summary:")
print("  Min:    ${:,.0f}".format(y.min()))
print("  Median: ${:,.0f}".format(np.median(y)))
print("  Max:    ${:,.0f}".format(y.max()))

# ── Baseline Ridge ────────────────────────────────────────────────────────
print("\nTraining Ridge baseline...")
ridge_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(ridge_pipeline, X_raw, y, cv=kf, scoring="r2")
print("  Ridge CV R2: {:.4f} (+/- {:.4f})".format(scores.mean(), scores.std()))
ridge_pipeline.fit(X_raw, y)

# ── XGBoost ───────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
)
cv_r2 = cross_val_score(xgb_model, X_clean, y, cv=kf, scoring="r2")
cv_mae = cross_val_score(xgb_model, X_clean, y, cv=kf, scoring="neg_mean_absolute_error")
print("  XGBoost CV R2:  {:.4f} (+/- {:.4f})".format(cv_r2.mean(), cv_r2.std()))
print("  XGBoost CV MAE: ${:,.0f}".format(-cv_mae.mean()))
xgb_model.fit(X_clean, y)
y_pred = xgb_model.predict(X_clean)

mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / np.where(y == 0, np.nan, y))) * 100
print("  MAE:  ${:,.0f}".format(mae))
print("  RMSE: ${:,.0f}".format(rmse))
print("  R2:   {:.4f}".format(r2))
print("  MAPE: {:.2f}%".format(mape))

# ── Save model artifacts ──────────────────────────────────────────────────
print("\nSaving model artifacts...")
joblib.dump(xgb_model, MODELS / "xgb_model.pkl")
joblib.dump(ridge_pipeline, MODELS / "ridge_baseline.pkl")
joblib.dump(imputer, MODELS / "imputer.pkl")
joblib.dump(feature_cols, MODELS / "feature_cols.pkl")
pd.DataFrame([{"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}]).to_csv(MODELS / "metrics.csv", index=False)
print("  Saved to models/")

# ── Feature importance ────────────────────────────────────────────────────
print("\nPlotting feature importance...")
importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 7))
importance.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 20 Feature Importances (XGBoost)", fontsize=14)
ax.set_xlabel("Importance Score")
plt.tight_layout()
fig.savefig(PLOTS / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved.")

# ── SHAP ──────────────────────────────────────────────────────────────────
print("\nComputing SHAP values (may take 1-2 min)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_clean)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_clean, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig(PLOTS / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  SHAP summary saved.")

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_clean, feature_names=feature_cols, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(PLOTS / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  SHAP bar saved.")

# ── Predictions vs actual ─────────────────────────────────────────────────
print("\nPlotting predictions vs actual...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y, y_pred, alpha=0.6, color="steelblue", edgecolors="none")
min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
ax.set_xlabel("Actual Median Home Value ($)", fontsize=12)
ax.set_ylabel("Predicted Median Home Value ($)", fontsize=12)
ax.set_title("Predicted vs Actual Home Values\nHillsborough County Census Tracts", fontsize=13)
ax.legend()
ax.text(0.05, 0.92, "R2 = {:.4f}".format(r2), transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.tight_layout()
fig.savefig(PLOTS / "predictions_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved.")

# ── Residuals ─────────────────────────────────────────────────────────────
print("Plotting residuals...")
residuals = y - y_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred, residuals, alpha=0.5, color="steelblue", edgecolors="none")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted Value ($)")
axes[0].set_ylabel("Residual ($)")
axes[0].set_title("Residuals vs Predicted")
axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("Residual ($)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")
plt.tight_layout()
fig.savefig(PLOTS / "residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved.")

# ── Save predictions GeoDataFrame ─────────────────────────────────────────
print("\nSaving predictions GeoDataFrame...")
gdf = gdf.copy()
gdf["predicted_home_value"] = y_pred
gdf["residual"] = gdf["target_home_value"] - gdf["predicted_home_value"]
gdf["pct_error"] = (gdf["residual"] / gdf["target_home_value"]) * 100
shap_df = pd.DataFrame(shap_values, columns=feature_cols)
gdf["top_shap_feature"] = shap_df.abs().idxmax(axis=1)
gdf["top_shap_value"] = shap_df.abs().max(axis=1)
gdf.to_file(PROCESSED / "predictions.geojson", driver="GeoJSON")
print("  predictions.geojson saved.")

print("\n" + "=" * 60)
print("ALL DONE!")
print("  R2:   {:.4f}".format(r2))
print("  MAE:  ${:,.0f}".format(mae))
print("  RMSE: ${:,.0f}".format(rmse))
print("  MAPE: {:.2f}%".format(mape))
print("=" * 60)
print("\nNext step: streamlit run src/streamlit_app.py")
