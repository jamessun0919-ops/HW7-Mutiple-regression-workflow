import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression, RFE
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import os

def main():
    print("=" * 60)
    print("  Feature Selection & Stepwise MSE Evaluation Workflow")
    print("=" * 60)

    # 1. Place/Load Data
    data_path = "data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"'{data_path}' not found. Please place your CSV file in the workspace directory.")

    df = pd.read_csv(data_path)
    print(f"Successfully loaded dataset with shape: {df.shape}")

    # 2. Target and Feature Auto-Detection
    target_candidates = ["medv", "Profit", "median_house_value"]
    target_col = None
    for cand in target_candidates:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    print(f"Target variable auto-detected: '{target_col}'")

    # Handle missing values in target column
    if df[target_col].isnull().sum() > 0:
        print(f"Dropping {df[target_col].isnull().sum()} rows with missing target values.")
        df = df.dropna(subset=[target_col])

    y = df[target_col].values
    X = df.drop(columns=[target_col])

    # 3. Column Type Identification
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numeric features: {len(numeric_cols)} -> {numeric_cols}")
    print(f"Categorical features: {len(categorical_cols)} -> {categorical_cols}")

    # 4. Data Preparation Pipeline
    transformers = []
    if len(numeric_cols) > 0:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_cols))

    if len(categorical_cols) > 0:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    print("Fitting and transforming data using ColumnTransformer...")
    X_preprocessed = preprocessor.fit_transform(X)

    # Reconstruct feature names
    feature_names = []
    if len(numeric_cols) > 0:
        feature_names.extend(numeric_cols)
    if len(categorical_cols) > 0:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_features)

    # Remove preprocessing prefixes if they exist (though get_feature_names_out is clean)
    feature_names = [f.split('__')[-1] for f in feature_names]

    X_df = pd.DataFrame(X_preprocessed, columns=feature_names)
    print(f"Preprocessed features shape: {X_df.shape}")
    print(f"Features list: {list(X_df.columns)}")

    # 5. Train-Test Split (80/20, seed 42)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    print("Train-Test split completed: 80% Train, 20% Test.")

    N_features = len(X_df.columns)

    # 6. Fit 9 Feature Selection Algorithms
    print("\n[Algorithm 1/9] Pearson Correlation...")
    pearson_corrs = [np.abs(np.corrcoef(X_train[col], y_train)[0, 1]) for col in X_train.columns]
    pearson_ranking = [x for _, x in sorted(zip(pearson_corrs, X_train.columns), reverse=True)]

    print("[Algorithm 2/9] Spearman Correlation...")
    spearman_corrs = []
    for col in X_train.columns:
        corr, _ = spearmanr(X_train[col], y_train)
        spearman_corrs.append(np.abs(corr) if not np.isnan(corr) else 0.0)
    spearman_ranking = [x for _, x in sorted(zip(spearman_corrs, X_train.columns), reverse=True)]

    print("[Algorithm 3/9] F-test Regression Score...")
    f_vals, _ = f_regression(X_train, y_train)
    # Replace nan/inf with 0
    f_vals = np.nan_to_num(f_vals)
    f_ranking = [x for _, x in sorted(zip(f_vals, X_train.columns), reverse=True)]

    print("[Algorithm 4/9] Mutual Information Score...")
    mi_vals = mutual_info_regression(X_train, y_train, random_state=42)
    mi_ranking = [x for _, x in sorted(zip(mi_vals, X_train.columns), reverse=True)]

    print("[Algorithm 5/9] Recursive Feature Elimination (RFE)...")
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
    rfe.fit(X_train, y_train)
    rfe_ranking = [x for _, x in sorted(zip(rfe.ranking_, X_train.columns))]

    print("[Algorithm 6/9] Sequential Forward Selection (SFS) custom loop...")
    selected_sfs = []
    remaining_sfs = list(X_train.columns)
    for _ in range(N_features):
        best_score = -np.inf
        best_feat = None
        for feat in remaining_sfs:
            candidate = selected_sfs + [feat]
            model = LinearRegression()
            model.fit(X_train[candidate], y_train)
            score = model.score(X_train[candidate], y_train)
            if score > best_score:
                best_score = score
                best_feat = feat
        selected_sfs.append(best_feat)
        remaining_sfs.remove(best_feat)
    sfs_ranking = selected_sfs

    print("[Algorithm 7/9] Sequential Backward Selection (SBS) custom loop...")
    current_sbs = list(X_train.columns)
    eliminated_sbs = []
    while len(current_sbs) > 1:
        best_score = -np.inf
        worst_feat = None
        for feat in current_sbs:
            candidate = [f for f in current_sbs if f != feat]
            model = LinearRegression()
            model.fit(X_train[candidate], y_train)
            score = model.score(X_train[candidate], y_train)
            if score > best_score:
                best_score = score
                worst_feat = feat
        eliminated_sbs.append(worst_feat)
        current_sbs.remove(worst_feat)
    eliminated_sbs.append(current_sbs[0])
    sbs_ranking = eliminated_sbs[::-1]

    print("[Algorithm 8/9] Lasso L1 Coefficients...")
    # Standardize target for better Lasso performance if needed, but LassoCV works fine on raw target
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
    lasso.fit(X_train, y_train)
    coef_abs = np.abs(lasso.coef_)
    # Use Pearson correlation as tie-breaker for zero coefficients
    sort_key = coef_abs + 1e-9 * np.array(pearson_corrs)
    lasso_ranking = [x for _, x in sorted(zip(sort_key, X_train.columns), reverse=True)]

    print("[Algorithm 9/9] Random Forest Feature Importances...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_
    rf_ranking = [x for _, x in sorted(zip(rf_importances, X_train.columns), reverse=True)]

    rankings = {
        "Pearson Corr": pearson_ranking,
        "Spearman Corr": spearman_ranking,
        "F-test Score": f_ranking,
        "Mutual Info": mi_ranking,
        "RFE (Linear)": rfe_ranking,
        "SFS (Forward)": sfs_ranking,
        "SBS (Backward)": sbs_ranking,
        "Lasso L1": lasso_ranking,
        "Random Forest": rf_ranking
    }

    # 7. Stepwise Modeling (from 1 to N features)
    print("\nRunning stepwise evaluations from k=1 to N...")
    stepwise_results = {}
    for name, ranking in rankings.items():
        mse_scores = []
        r2_scores = []
        for k in range(1, N_features + 1):
            top_k = ranking[:k]
            model = LinearRegression()
            model.fit(X_train[top_k], y_train)
            preds = model.predict(X_test[top_k])
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            mse_scores.append(mse)
            r2_scores.append(r2)
        stepwise_results[name] = {
            "mse": mse_scores,
            "r2": r2_scores
        }

    # Calculate Frontier Paths (ceiling R2 and floor MSE at each k)
    k_range = list(range(1, N_features + 1))
    frontier_mse = []
    frontier_r2 = []
    for idx in range(N_features):
        k_mse_vals = [stepwise_results[name]["mse"][idx] for name in rankings]
        k_r2_vals = [stepwise_results[name]["r2"][idx] for name in rankings]
        frontier_mse.append(min(k_mse_vals))
        frontier_r2.append(max(k_r2_vals))

    # Calculate Sweet Point (Elbow Point where R2 reaches at least 97% of max R2)
    max_r2_val = max(frontier_r2)
    sweet_k = 1
    for k in range(1, N_features + 1):
        if frontier_r2[k-1] >= 0.97 * max_r2_val:
            sweet_k = k
            break
    sweet_k = max(1, min(sweet_k, N_features))
    print(f"Calculated Sweet Point (k={sweet_k} features, R2={frontier_r2[sweet_k-1]:.4f})")

    # Create Rank Table DataFrame
    table_data = {"Rank": k_range}
    for name in rankings:
        table_data[name] = rankings[name]
    df_table = pd.DataFrame(table_data)

    # 8. Unified Visualization
    print("\nGenerating unified visualization...")
    fig = plt.figure(figsize=(16, 18), facecolor="#f8fafc")
    
    # 2 Subplots side-by-side for curves
    ax_r2 = fig.add_subplot(2, 2, 1)
    ax_mse = fig.add_subplot(2, 2, 2)
    # A wide subplot at the bottom for the table
    ax_table = fig.add_subplot(2, 1, 2)

    # Color Palette for 9 algorithms
    colors = {
        "Pearson Corr": "#ef4444",      # Red
        "Spearman Corr": "#f97316",     # Orange
        "F-test Score": "#eab308",      # Yellow
        "Mutual Info": "#22c55e",       # Green
        "RFE (Linear)": "#06b6d4",      # Cyan
        "SFS (Forward)": "#3b82f6",     # Blue
        "SBS (Backward)": "#6366f1",    # Indigo
        "Lasso L1": "#8b5cf6",          # Purple
        "Random Forest": "#ec4899"      # Pink
    }

    # Plot R-squared Curves
    ax_r2.set_facecolor("#ffffff")
    ax_r2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    for name in rankings:
        ax_r2.plot(k_range, stepwise_results[name]["r2"], label=name, color=colors[name], linewidth=2.0, marker="o", markersize=4)
    ax_r2.plot(k_range, frontier_r2, label="Frontier Path", color="#f97316", linewidth=2.5, linestyle=":", marker="D", markersize=5)
    
    # Mark Sweet Point
    ax_r2.axvline(x=sweet_k, color="#ea580c", linestyle="--", alpha=0.7, linewidth=1.5)
    ax_r2.scatter(sweet_k, frontier_r2[sweet_k-1], color="#ea580c", s=120, zorder=10, edgecolor="black", linewidth=1.5)
    
    # Dynamic text positioning for R2
    x_text_r2 = sweet_k - 2.5 if sweet_k > 3 else sweet_k + 0.5
    y_text_r2 = frontier_r2[sweet_k-1] - 0.08 if frontier_r2[sweet_k-1] > 0.2 else frontier_r2[sweet_k-1] + 0.05
    ax_r2.annotate(f"Sweet Point (k={sweet_k})", 
                    xy=(sweet_k, frontier_r2[sweet_k-1]), 
                    xytext=(x_text_r2, y_text_r2),
                    arrowprops=dict(facecolor="#ea580c", shrink=0.08, width=1.5, headwidth=6, headlength=6),
                    fontweight="bold", color="#ea580c", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff7ed", ec="#ffedd5", alpha=0.9))

    ax_r2.set_title("Test R-squared vs. Feature Subset Size (k)", fontsize=12, fontweight="bold", color="#1e293b", pad=12)
    ax_r2.set_xlabel("Feature Subset Size (k)", fontsize=10, color="#475569")
    ax_r2.set_ylabel("Test R-squared", fontsize=10, color="#475569")
    ax_r2.set_xticks(k_range)
    ax_r2.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax_r2.spines["top"].set_visible(False)
    ax_r2.spines["right"].set_visible(False)
    ax_r2.spines["left"].set_color("#cbd5e1")
    ax_r2.spines["bottom"].set_color("#cbd5e1")

    # Plot MSE Curves
    ax_mse.set_facecolor("#ffffff")
    ax_mse.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    for name in rankings:
        ax_mse.plot(k_range, stepwise_results[name]["mse"], label=name, color=colors[name], linewidth=2.0, marker="o", markersize=4)
    ax_mse.plot(k_range, frontier_mse, label="Frontier Path", color="#f97316", linewidth=2.5, linestyle=":", marker="D", markersize=5)
    
    # Mark Sweet Point
    ax_mse.axvline(x=sweet_k, color="#ea580c", linestyle="--", alpha=0.7, linewidth=1.5)
    ax_mse.scatter(sweet_k, frontier_mse[sweet_k-1], color="#ea580c", s=120, zorder=10, edgecolor="black", linewidth=1.5)
    
    # Dynamic text positioning for MSE
    x_text_mse = sweet_k - 2.5 if sweet_k > 3 else sweet_k + 0.5
    y_text_mse = frontier_mse[sweet_k-1] + (max(frontier_mse) - min(frontier_mse)) * 0.15
    ax_mse.annotate(f"Sweet Point (k={sweet_k})", 
                    xy=(sweet_k, frontier_mse[sweet_k-1]), 
                    xytext=(x_text_mse, y_text_mse),
                    arrowprops=dict(facecolor="#ea580c", shrink=0.08, width=1.5, headwidth=6, headlength=6),
                    fontweight="bold", color="#ea580c", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fff7ed", ec="#ffedd5", alpha=0.9))

    ax_mse.set_title("Test MSE vs. Feature Subset Size (k)", fontsize=12, fontweight="bold", color="#1e293b", pad=12)
    ax_mse.set_xlabel("Feature Subset Size (k)", fontsize=10, color="#475569")
    ax_mse.set_ylabel("Test Mean Squared Error (MSE)", fontsize=10, color="#475569")
    ax_mse.set_xticks(k_range)
    ax_mse.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax_mse.spines["top"].set_visible(False)
    ax_mse.spines["right"].set_visible(False)
    ax_mse.spines["left"].set_color("#cbd5e1")
    ax_mse.spines["bottom"].set_color("#cbd5e1")

    # Custom Table Drawer
    def draw_premium_table(ax, df_table):
        ax.axis("off")
        cols = df_table.columns
        nrows = len(df_table)
        ncols = len(cols)
        
        # Define variable column widths (Rank is narrower, algorithms are wider)
        widths = [0.8] + [1.8] * (ncols - 1)
        total_width = sum(widths)
        
        # Header
        for c_idx, col in enumerate(cols):
            w = widths[c_idx]
            x = sum(widths[:c_idx])
            # Header background patch
            rect = plt.Rectangle((x, nrows), w, 1, facecolor="#1e293b", edgecolor="#cbd5e1", linewidth=0.7)
            ax.add_patch(rect)
            ax.text(x + w / 2, nrows + 0.5, col, ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            
        # Rows
        for r_idx in range(nrows):
            # Zebra striping
            row_color = "#f8fafc" if r_idx % 2 == 0 else "#ffffff"
            for c_idx in range(ncols):
                w = widths[c_idx]
                x = sum(widths[:c_idx])
                val = df_table.iloc[r_idx, c_idx]
                
                # Cell background patch
                rect = plt.Rectangle((x, nrows - 1 - r_idx), w, 1, facecolor=row_color, edgecolor="#cbd5e1", linewidth=0.5)
                ax.add_patch(rect)
                
                # Check for Rank column styling
                if c_idx == 0:
                    ax.text(x + w / 2, nrows - 1 - r_idx + 0.5, str(val), ha="center", va="center", color="#1e293b", fontweight="bold", fontsize=9)
                else:
                    # Feature cell styling
                    ax.text(x + w / 2, nrows - 1 - r_idx + 0.5, str(val), ha="center", va="center", color="#334155", fontsize=8)
                    
        ax.set_xlim(0, total_width)
        ax.set_ylim(-0.5, nrows + 1.5)

    draw_premium_table(ax_table, df_table)
    
    # Dynamic Dataset Name Mapping
    dataset_name = "Custom Tabular Dataset"
    if target_col == "medv":
        dataset_name = "Boston Housing"
    elif target_col == "median_house_value":
        dataset_name = "California Housing"
    elif target_col == "Profit":
        dataset_name = "50 Startups"

    # Titles and spacing
    fig.suptitle(f"General Feature Selection & Stepwise MSE Evaluation\nDataset: {dataset_name} | Target: {target_col}", 
                 fontsize=16, fontweight="bold", color="#0f172a", y=0.96)
    
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.06, right=0.94, hspace=0.3, wspace=0.2)
    
    output_img = "feature_selection_performance_allinone.png"
    plt.savefig(output_img, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    
    print(f"\nSuccessfully generated and saved publication-quality plot to: '{output_img}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
