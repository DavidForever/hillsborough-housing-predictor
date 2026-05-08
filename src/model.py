"""
model.py
Trains an XGBoost model to predict median home values by census tract
in Hillsborough County, FL. Includes cross-validation, SHAP explainability,
and saves the trained model and evaluation outputs.
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


# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PLOTS = ROOT / "outputs" / "plots"

for p in [MODELS, PLOTS]:
    p.mkdir(parents=True, exist_ok=True)


def load_features():
    """Load the feature matrix from disk."""
    print("Loading feature matrix...")
    csv_path = PROCESSED / "features.csv"
    gdf_path = PROCESSED / "features.geojson"
    df = pd.read_csv(csv_path)
    gdf = gpd.read_file(gdf_path)
    print("  Loaded {} tracts with {} columns".format(len(df), len(df.columns)))
    return df, gdf


def get_feature_cols(df):
    """Return the list of feature columns (exclude GEOID and target)."""
    exclude = ["GEOID", "target_home_value", "geometry"]
    cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
    return cols


def evaluate_model(y_true, y_pred, label=""):
    """Print regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    print("  {} Metrics:".format(label))
    print("    MAE:  ${:,.0f}".format(mae))
    print("    RMSE: ${:,.0f}".format(rmse))
    print("    R2:   {:.4f}".format(r2))
    print("    MAPE: {:.2f}%".format(mape))
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def train_baseline(X_raw, y):
    """Train a Ridge regression baseline with imputation and scaling pipeline."""
    print("\nTraining baseline Ridge regression...")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_raw, y, cv=kf, scoring="r2")
    print("  Ridge CV R2: {:.4f} (+/- {:.4f})".format(scores.mean(), scores.std()))
    pipeline.fit(X_raw, y)
    return pipeline


def train_xgboost(X_clean, y):
    """Train XGBoost model with cross-validation."""
    print("\nTraining XGBoost model...")
    model = XGBRegressor(
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
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X_clean, y, cv=kf, scoring="r2")
    cv_mae = cross_val_score(model, X_clean, y, cv=kf, scoring="neg_mean_absolute_error")
    print("  XGBoost CV R2:  {:.4f} (+/- {:.4f})".format(cv_r2.mean(), cv_r2.std()))
    print("  XGBoost CV MAE: ${:,.0f} (+/- ${:,.0f})".format(-cv_mae.mean(), cv_mae.std()))
    model.fit(X_clean, y)
    print("  XGBoost training complete.")
    return model


def plot_feature_importance(model, feature_cols):
    """Plot XGBoost built-in feature importance."""
    print("\nPlotting feature importance...")
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    importance.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 20 Feature Importances (XGBoost)", fontsize=14)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    out = PLOTS / "feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved ->", out)


def plot_shap_values(model, X_clean, feature_cols):
    """Generate SHAP summary and bar plots."""
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_clean, feature_names=feature_cols, show=False)
    plt.tight_layout()
    out = PLOTS / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP summary saved ->", out)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_clean, feature_names=feature_cols, plot_type="bar", show=False)
    plt.tight_layout()
    out2 = PLOTS / "shap_bar.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP bar plot saved ->", out2)

    return shap_values


def plot_predictions_vs_actual(y_true, y_pred):
    """Scatter plot of predicted vs actual home values."""
    print("\nPlotting predictions vs actual...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="none", color="steelblue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual Median Home Value ($)", fontsize=12)
    ax.set_ylabel("Predicted Median Home Value ($)", fontsize=12)
    ax.set_title("Predicted vs Actual Home Values\nHillsborough County Census Tracts", fontsize=13)
    ax.legend()
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, "R2 = {:.4f}".format(r2), transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    out = PLOTS / "predictions_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved ->", out)


def plot_residuals(y_true, y_pred):
    """Plot residual distribution."""
    print("Plotting residuals...")
    residuals = y_true - y_pred
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
    out = PLOTS / "residuals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved ->", out)


def save_predictions_to_geodataframe(gdf, y_pred, shap_values, feature_cols):
    """Attach predictions and top SHAP driver to the GeoDataFrame and save."""
    print("\nSaving predictions to GeoDataFrame...")
    gdf = gdf.copy()
    gdf["predicted_home_value"] = y_pred
    gdf["residual"] = gdf["target_home_value"] - gdf["predicted_home_value"]
    gdf["pct_error"] = (gdf["residual"] / gdf["target_home_value"]) * 100
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    gdf["top_shap_feature"] = shap_df.abs().idxmax(axis=1)
    gdf["top_shap_value"] = shap_df.abs().max(axis=1)
    out = PROCESSED / "predictions.geojson"
    gdf.to_file(out, driver="GeoJSON")
    print("  Predictions saved ->", out)
    return gdf


def run_modeling():
    """Full modeling pipeline."""
    print("=" * 60)
    print("HILLSBOROUGH HOUSING PREDICTOR -- Model Training")
    print("=" * 60)

    df, gdf = load_features()
    feature_cols = get_feature_cols(df)
    X_raw = df[feature_cols].values
    y = df["target_home_value"]

    # Report and impute NaNs
    nan_counts = pd.DataFrame(X_raw, columns=feature_cols).isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print("\nImputing NaN values in {} columns with median:".format(len(nan_cols)))
        for col, count in nan_cols.items():
            print("  {}: {} NaN(s)".format(col, count))

    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X_raw)
    print("\nImputation complete. X shape: {}".format(X_clean.shape))

    print("\nTarget variable summary:")
    print("  Min:    ${:,.0f}".format(y.min()))
    print("  Median: ${:,.0f}".format(y.median()))
    print("  Max:    ${:,.0f}".format(y.max()))
    print("  Features used: {}".format(len(feature_cols)))

    # Train
    ridge_pipeline = train_baseline(X_raw, y)
    ridge_preds = ridge_pipeline.predict(X_raw)
    evaluate_model(y, ridge_preds, label="Ridge Baseline (train)")

    xgb_model = train_xgboost(X_clean, y)
    xgb_preds = xgb_model.predict(X_clean)
    metrics = evaluate_model(y, xgb_preds, label="XGBoost (train)")

    # Plots
    plot_feature_importance(xgb_model, feature_cols)
    shap_values = plot_shap_values(xgb_model, X_clean, feature_cols)
    plot_predictions_vs_actual(y, xgb_preds)
    plot_residuals(y, xgb_preds)

    # Save
    pred_gdf = save_predictions_to_geodataframe(gdf, xgb_preds, shap_values, feature_cols)

    print("\nSaving model artifacts...")
    joblib.dump(xgb_model, MODELS / "xgb_model.pkl")
    joblib.dump(ridge_pipeline, MODELS / "ridge_baseline.pkl")
    joblib.dump(imputer, MODELS / "imputer.pkl")
    joblib.dump(feature_cols, MODELS / "feature_cols.pkl")
    pd.DataFrame([metrics]).to_csv(MODELS / "metrics.csv", index=False)
    print("  Models saved to:", MODELS)

    print("\n" + "=" * 60)
    print("Modeling complete!")
    print("  R2:   {:.4f}".format(metrics["r2"]))
    print("  MAE:  ${:,.0f}".format(metrics["mae"]))
    print("  RMSE: ${:,.0f}".format(metrics["rmse"]))
    print("  MAPE: {:.2f}%".format(metrics["mape"]))
    print("=" * 60)

    return xgb_model, pred_gdf, feature_cols, shap_values


if __name__ == "__main__":
    run_modeling()
