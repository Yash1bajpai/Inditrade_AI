"""
Trade AI Evaluation & Visualization Suite
Phase 3: Quantitative Model Analytics (Module A, C, D)
Generates publication-grade evaluation plots across XGBoost, Isolation Forest, and Node2Vec embeddings.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = '#121212'
plt.rcParams['axes.facecolor'] = '#1E1E1E'
plt.rcParams['text.color'] = '#E0E0E0'
plt.rcParams['axes.labelcolor'] = '#E0E0E0'
plt.rcParams['xtick.color'] = '#B0B0B0'
plt.rcParams['ytick.color'] = '#B0B0B0'
plt.rcParams['grid.color'] = '#2A2A2A'

def compute_smape(y_true, y_pred):
    """Compute Symmetric Mean Absolute Percentage Error (SMAPE)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > 1e-6
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100.0

def main():
    print("=== [PHASE 3 VISUALIZATION] QUANTITATIVE MODEL EVALUATION SUITE ===")
    os.makedirs("reports/figures", exist_ok=True)
    
    # 1. Load data & models
    print("[*] Loading processed trade features and trained checkpoints...")
    df = pd.read_parquet("data/processed/trade_features.parquet")
    
    # Load XGBoost checkpoint dict
    xgb_data = joblib.load("models/xgboost_trade_forecast.pkl")
    xgb_model = xgb_data["model"]
    feature_cols = xgb_data["features"]
    
    # Prepare X and y with exact float32 numerical casting
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(np.float32)
    X = X.replace([np.inf, -np.inf], 0).fillna(0)
    
    y_raw = df["primaryValue"].values
    y_log_true = np.log1p(np.maximum(y_raw, 0))
    
    # Predict
    print("[*] Generating XGBoost trade flow predictions across dataset...")
    y_log_pred = xgb_model.predict(X)
    y_raw_pred = np.expm1(np.maximum(y_log_pred, 0))
    
    # Calculate comprehensive metrics
    log_rmse = np.sqrt(mean_squared_error(y_log_true, y_log_pred))
    log_r2 = r2_score(y_log_true, y_log_pred)
    dollar_rmse = np.sqrt(mean_squared_error(y_raw, y_raw_pred))
    dollar_mae = mean_absolute_error(y_raw, y_raw_pred)
    smape_val = compute_smape(y_raw, y_raw_pred)
    
    print("\n=== XGBOOST TRADE FORECAST METRICS SUMMARY ===")
    print(f"  * Log-Scale R² Score   : {log_r2:.4f}")
    print(f"  * Log-Scale RMSE       : {log_rmse:.4f}")
    print(f"  * Dollar-Scale RMSE    : ${dollar_rmse:,.2f}")
    print(f"  * Dollar-Scale MAE     : ${dollar_mae:,.2f}")
    print(f"  * Dollar-Scale SMAPE   : {smape_val:.2f}%\n")
    
    # -------------------------------------------------------------------------
    # PLOT 1: XGBoost Actual vs Predicted Trade Flows (Log Scale & Dollar Scale)
    # -------------------------------------------------------------------------
    print("[*] Generating Plot 1: Actual vs Predicted Trade Flow Trajectory...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # Subplot 1: Log-Scale Correlation
    sns.scatterplot(x=y_log_true, y=y_log_pred, alpha=0.3, color='#00E676', s=15, ax=ax1)
    ax1.plot([0, 26], [0, 26], color='#FF5252', linestyle='--', linewidth=2, label="Ideal Fit (y = x)")
    ax1.set_title("Actual vs Predicted Trade Value (Log-Scale log1p)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Actual Log Trade Value", fontsize=11)
    ax1.set_ylabel("Predicted Log Trade Value", fontsize=11)
    ax1.text(0.05, 0.88, f"Log R² = {log_r2:.4f}\nLog RMSE = {log_rmse:.4f}\nSMAPE = {smape_val:.1f}%", 
             transform=ax1.transAxes, fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='#2A2A2A', edgecolor='#00E676'))
    ax1.legend(loc="lower right")
    ax1.set_xlim(-0.5, 26.5)
    ax1.set_ylim(-0.5, 26.5)
    
    # Subplot 2: Residual Distribution
    residuals = y_log_true - y_log_pred
    sns.histplot(residuals, bins=60, kde=True, color='#29B6F6', ax=ax2)
    ax2.axvline(0, color='#FF5252', linestyle='--', linewidth=2, label="Zero Error")
    ax2.set_title("Prediction Residual Density (Log-Scale Error)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Residual (Actual log - Predicted log)", fontsize=11)
    ax2.set_ylabel("Trade Flow Count", fontsize=11)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("reports/figures/01_xgboost_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[Exported] -> reports/figures/01_xgboost_actual_vs_predicted.png")
    
    # -------------------------------------------------------------------------
    # PLOT 2: XGBoost Top 15 Feature Importances
    # -------------------------------------------------------------------------
    print("[*] Generating Plot 2: Top 15 Feature Importance Bar Chart...")
    importances = xgb_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False).head(15)
    
    plt.figure(figsize=(12, 7), dpi=300)
    ax = sns.barplot(x="Importance", y="Feature", data=feat_df, palette="crest")
    plt.title("Top 15 Most Influential Features in Trade Flow Forecasting (XGBoost Module A)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Gini Importance Score (Normalized)", fontsize=12)
    plt.ylabel("Engineered Feature", fontsize=12)
    
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.005, p.get_y() + p.get_height()/2.0, f'{width:.4f}', 
                ha='left', va='center', fontsize=10, color='#00E676', fontweight='bold')
        
    plt.xlim(0, max(feat_df["Importance"]) * 1.15)
    plt.tight_layout()
    plt.savefig("reports/figures/02_xgboost_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[Exported] -> reports/figures/02_xgboost_feature_importance.png")
    
    # -------------------------------------------------------------------------
    # PLOT 3: Isolation Forest Trade Anomaly Distribution
    # -------------------------------------------------------------------------
    print("[*] Generating Plot 3: Trade Anomaly Score Density & Extreme Shocks...")
    # Load anomaly detector
    iso_data = joblib.load("models/isolation_forest_anomalies.pkl")
    iso_model = iso_data["model"] if isinstance(iso_data, dict) else iso_data
    # Compute engineered anomaly indicators matching load_anomaly_features
    df['unit_value'] = df['primaryValue'] / np.maximum(df['netWgt'].fillna(0), 1.0)
    df['value_vs_3y_mean'] = df['primaryValue'] / np.maximum(df['primaryValue_rolling_3y_mean'].fillna(0), 1.0)
    df['wgt_vs_3y_mean'] = df['netWgt'] / np.maximum(df['netWgt_rolling_3y_mean'].fillna(0), 1.0)
    
    anomaly_indicators = [
        'primaryValue_yoy_growth_rate', 'value_vs_3y_mean', 'wgt_vs_3y_mean',
        'unit_value', 'usdinr_yoy_pct', 'brent_crude_yoy_pct', 'policy_event_flag'
    ]
    cols_present = [c for c in anomaly_indicators if c in df.columns]
    
    # Ensure feature cols exist and fillna
    X_anom = df[cols_present].copy()
    for col in X_anom.columns:
        X_anom[col] = pd.to_numeric(X_anom[col], errors='coerce').fillna(0).astype(np.float32)
    X_anom = X_anom.replace([np.inf, -np.inf], 0).fillna(0)
        
    raw_scores = iso_model.decision_function(X_anom)
    anomaly_scores = -raw_scores
    df["anomaly_score"] = anomaly_scores
    
    # Calculate threshold (99th percentile for 1% contamination)
    threshold = np.percentile(anomaly_scores, 99.0)
    
    plt.figure(figsize=(14, 7), dpi=300)
    sns.histplot(anomaly_scores, bins=80, kde=True, color='#AB47BC')
    plt.axvline(threshold, color='#FF5252', linestyle='--', linewidth=2.5, label=f"Anomaly Cutoff Threshold (Score: {threshold:.4f} | Top 1%)")
    
    # Annotate Top 2 anomalies
    top_anoms = df.sort_values("anomaly_score", ascending=False).head(2)
    for idx, row in top_anoms.iterrows():
        p_name = str(row.get('partnerDesc', row.get('partnerCode', 'N/A')))[:12]
        c_code = row.get('cmdCode', 'N/A')
        score = row['anomaly_score']
        plt.annotate(f"Shock: {p_name} HS{c_code}\nScore: {score:.4f}\nYoY: +{row.get('primaryValue_yoy_growth_rate',0):,.0f}%",
                     xy=(score, 50), xytext=(score - 0.015, 600),
                     arrowprops=dict(facecolor='#FFD54F', shrink=0.05, width=1.5, headwidth=6),
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor='#2A2A2A', edgecolor='#FFD54F'))
        
    plt.title("Distribution of Bilateral Trade Flow Anomaly Scores (Isolation Forest Module D)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Anomaly Score (-Decision Function)", fontsize=12)
    plt.ylabel("Number of Trade Flow Records", fontsize=12)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/figures/03_isolation_forest_anomaly_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[Exported] -> reports/figures/03_isolation_forest_anomaly_distribution.png")
    
    # -------------------------------------------------------------------------
    # PLOT 4: Node2Vec Trade Network 2D Projection
    # -------------------------------------------------------------------------
    print("[*] Generating Plot 4: Node2Vec Bilateral Trade Graph Embedding Space...")
    nodes_df = pd.read_parquet("data/processed/graph_nodes.parquet")
    embeddings_df = pd.read_parquet("data/processed/node2vec_trade_embeddings.parquet")
    
    # Merge embeddings with node metadata
    merged_nodes = pd.merge(nodes_df, embeddings_df[["node_id", "embedding_vector"]], left_on="node_id", right_on="node_id", how="inner")
    
    emb_matrix = np.stack(merged_nodes['embedding_vector'].values)
    
    # Reduce to 2D via PCA
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(emb_matrix)
    merged_nodes["pca_x"] = emb_2d[:, 0]
    merged_nodes["pca_y"] = emb_2d[:, 1]
    
    plt.figure(figsize=(14, 9), dpi=300)
    sns.scatterplot(
        x="pca_x", y="pca_y", hue="node_type", style="node_type",
        data=merged_nodes, palette={"commodity": "#26A69A", "partner": "#FFCA28"},
        s=120, alpha=0.85
    )
    
    # Annotate all partner countries and top commodity nodes
    for idx, row in merged_nodes.iterrows():
        if row["node_type"] == "partner":
            label = str(row.get("node_desc", row["node_id"]))[:15]
            plt.text(row["pca_x"] + 0.02, row["pca_y"] + 0.02, label, fontsize=9, color='#FFCA28', fontweight='bold')
        elif row["node_type"] == "commodity" and str(row["node_id"]) in ["C_27", "C_85", "C_84", "C_71", "C_30", "C_88"]:
            label = f"HS {row['node_id'].replace('C_','')}"
            plt.text(row["pca_x"] + 0.02, row["pca_y"] - 0.03, label, fontsize=8.5, color='#80CBC4')
            
    plt.title("2D Structural Projection of Bilateral Trade Network (Node2Vec Module C PCA)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f"PCA Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)", fontsize=12)
    plt.ylabel(f"PCA Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)", fontsize=12)
    plt.legend(title="Graph Node Type", loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/figures/04_node2vec_trade_network_clusters.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[Exported] -> reports/figures/04_node2vec_trade_network_clusters.png")
    
    # -------------------------------------------------------------------------
    # Plot 5: Feature Ablation Study - Macro/Forex Importance Surge (Grouped Bar)
    # -------------------------------------------------------------------------
    print("[*] Generating Plot 5: Feature Ablation Study - Macro/Forex Importance Surge...")
    ablation_sets = [
        "Set 1: Full Baseline\n(43 Features)",
        "Set 2: Without Top 3 Lags\n(Drop rolling 3y/5y, lag 1y)",
        "Set 3: Without ALL Lags\n(Drop all autoregressive)"
    ]
    lag_pcts = [98.77, 39.81, 3.53]
    struct_pcts = [0.38, 18.02, 70.38]
    macro_pcts = [1.27, 44.11, 26.09]
    
    x = np.arange(len(ablation_sets))
    width = 0.26
    
    plt.figure(figsize=(11, 6.5))
    plt.bar(x - width, lag_pcts, width, label='Lag / Rolling Indicators (Autoregressive)', color='#EF5350', alpha=0.9, edgecolor='#B71C1C')
    plt.bar(x, struct_pcts, width, label='Structural / Identifiers (cmdCode, partnerCode)', color='#42A5F5', alpha=0.9, edgecolor='#0D47A1')
    plt.bar(x + width, macro_pcts, width, label='Macro / Forex / Policy Shocks (USD, GBP, Oil, Gold)', color='#26A69A', alpha=0.95, edgecolor='#004D40', hatch='//')
    
    for i in range(len(ablation_sets)):
        plt.text(x[i] - width, lag_pcts[i] + 1.5, f"{lag_pcts[i]:.1f}%", ha='center', va='bottom', color='#FFCDD2', fontsize=9.5, fontweight='bold')
        plt.text(x[i], struct_pcts[i] + 1.5, f"{struct_pcts[i]:.1f}%", ha='center', va='bottom', color='#BBDEFB', fontsize=9.5, fontweight='bold')
        plt.text(x[i] + width, macro_pcts[i] + 1.5, f"{macro_pcts[i]:.1f}%", ha='center', va='bottom', color='#A7FFEB', fontsize=10.5, fontweight='bold')
        
    plt.title("Empirical Feature Ablation: Co-linearity Shadowing & Macro Feature Surge", fontsize=14, fontweight='bold', pad=18)
    plt.xlabel("Experimental Feature Set Configuration (5-Fold TimeSeriesSplit across 32,072 rows)", fontsize=12, labelpad=10)
    plt.ylabel("Relative XGBoost Split Gain Share (%)", fontsize=12)
    plt.xticks(x, ablation_sets, fontsize=11, fontweight='bold')
    plt.ylim(0, 115)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=1, frameon=True, fontsize=10.5)
    plt.tight_layout()
    plt.savefig("reports/figures/05_feature_ablation_macro_surge.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[Exported] -> reports/figures/05_feature_ablation_macro_surge.png")
    
    # -------------------------------------------------------------------------
    # Plot 6: Macro & Forex Feature Explanatory Power Deep Dive (Unshadowed)
    # -------------------------------------------------------------------------
    print("[*] Generating Plot 6: Macro & Forex Feature Deep Dive (Unshadowed Explanatory Power)...")
    macro_features = [
        'gbpinr_year_end\n(GBP/INR Exchange Rate)',
        'jpyinr_year_end\n(JPY/INR Exchange Rate)',
        'eurinr_mean\n(EUR/INR Average Rate)',
        'brent_crude_mean\n(Global Oil Shocks)',
        'nifty_50_mean\n(Indian Stock Benchmark)',
        'usdinr_vol_std\n(USD/INR Volatility)',
        'gold_futures_yoy_pct\n(Precious Metal Cycles)',
        'policy_event_flag\n(DGFT/PIB Interventions)'
    ]
    # Gain % values when unshadowed (Set 2 & Set 3 maximum active gain)
    macro_gains = [16.97, 6.96, 6.95, 6.52, 3.36, 3.40, 2.94, 0.22]
    
    df_macro_plot = pd.DataFrame({'feature': macro_features, 'gain_pct': macro_gains}).sort_values(by='gain_pct', ascending=True)
    
    plt.figure(figsize=(11, 6))
    colors_macro = ['#80CBC4' if g < 5.0 else '#26A69A' if g < 10.0 else '#00897B' for g in df_macro_plot['gain_pct']]
    bars = sns.barplot(x="gain_pct", y="feature", data=df_macro_plot, palette=colors_macro, hue="feature", legend=False)
    
    for p in bars.patches:
        width_val = p.get_width()
        if width_val > 0:
            plt.text(width_val + 0.3, p.get_y() + p.get_height() / 2, f"{width_val:.2f}%", 
                     va='center', color='#E0F2F1', fontweight='bold', fontsize=10.5)
            
    plt.title("Top Macroeconomic & Forex Indicators When Unshadowed by Lags (Gain %)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Relative XGBoost Explanatory Share (% of Total Split Gain in Ablation Sets)", fontsize=12)
    plt.ylabel("Macro / Forex Indicator & Economic Interpretation", fontsize=12)
    plt.xlim(0, max(df_macro_plot['gain_pct']) * 1.18)
    plt.tight_layout()
    plt.savefig("reports/figures/06_macro_feature_deep_dive.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[Exported] -> reports/figures/06_macro_feature_deep_dive.png")
    
    print("\n[SUCCESS] All 6 publication-grade evaluation plots successfully generated in reports/figures/!")

if __name__ == "__main__":
    main()
