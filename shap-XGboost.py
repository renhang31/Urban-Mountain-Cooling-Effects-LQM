import matplotlib

matplotlib.use('Agg')  # Solve PyCharm display issue

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import shap
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, entropy
import joblib
import os
from scipy import stats
from scipy.stats import gaussian_kde

# ========== Scientific Plot Global Settings ==========
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'font.size': 18,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})
colors = ['#2E75B6', '#ED7D31', '#A5A5A5', '#FFC000', '#4472C4']

# ========== Data Preprocessing ==========
# Simplified data file path (assuming script and data are in the same directory)
data_file = '2023/resampledclip_data_sorted_standardized.csv'#Change your years
data = pd.read_csv(data_file)

# Separate features and target variable (MCI is the target)
cols = [col for col in data.columns if col != 'MCI']
X_train, X_test, y_train, y_test = train_test_split(data[cols], data['MCI'], test_size=0.2, random_state=42)

# ========== Model Training ==========
model = xgb.XGBRegressor(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=150,
    random_state=42,
    reg_alpha=0.1,
    reg_lambda=0.1
)
model.fit(X_train, y_train)

# ========== Model Evaluation ==========
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
pearson_r, _ = pearsonr(y_test, y_pred)


# Calculate JSD (Jensen-Shannon Divergence)
def jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


# Simple binning to calculate JSD between predicted and true values
hist_true, _ = np.histogram(y_test, bins=10, density=True)
hist_pred, _ = np.histogram(y_pred, bins=10, density=True)
jsd_value = jsd(hist_true, hist_pred)

print(
    f'MSE: {mean_squared_error(y_test, y_pred):.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nPearson\'s r: {pearson_r:.4f}\nJSD: {jsd_value:.4f}')


# ========== Validation Visualization Module ==========
def enhanced_validation_plots(y_true, y_pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    residuals = y_pred - y_true

    # Construct simplified metrics text
    metrics_text = (
        f'RMSE = {rmse:.4f}\n'
        f'R² = {r2:.4f}\n'
        f'Pearson\'s r = {pearson_r:.4f}\n'
        f'JSD = {jsd_value:.4f}'
    )
    # Residual analysis plot
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(residuals, kde=True, color=colors[0], ax=ax[0], bins=15,
                 edgecolor='white', linewidth=1.2)
    ax[0].set(xlabel='Residuals', ylabel='Density', title='Residual Distribution')

    stats.probplot(residuals, dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set(markerfacecolor=colors[1], markersize=8)
    ax[1].set_title('Normal Q-Q Plot')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_analysis.png'))
    plt.close()


# ========== Execute Validation Plotting ==========
# Simplified output directory path
output_dir = 'XGB2023'
enhanced_validation_plots(y_test, y_pred, output_dir)

# ========== SHAP Analysis ==========
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
os.makedirs(output_dir, exist_ok=True)


# ========== Visualization Enhancement Module ==========
def sci_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')  # Set x-axis to black
    ax.tick_params(width=1.5, length=6)


# SHAP summary plot
fig = plt.figure(figsize=(8, 20))
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
sci_style(plt.gca())
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'summary_plot.png'))
plt.close()

# Feature importance data
shap_df = pd.DataFrame({
    'Feature': cols,
    'Importance': np.abs(shap_values.values).mean(axis=0)
}).sort_values('Importance', ascending=False)
shap_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

# Model feature importance ranking plot
plt.figure(figsize=(8, 16))
sns.barplot(x='Importance', y='Feature', data=shap_df, palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Ranking')
sci_style(plt.gca())
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_ranking.png'))
plt.close()

# SHAP dependence plots
for idx, feature in enumerate(cols):
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(idx, shap_values.values, X_test, feature_names=cols, ax=ax)
    sci_style(ax)
    plt.savefig(os.path.join(output_dir, f'dependence_{feature}.png'))
    plt.close()

# SHAP waterfall plot
fig = plt.figure(figsize=(8, 20))
shap.plots.waterfall(shap_values[30], max_display=23)
sci_style(plt.gca())
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'waterfall_plot.png'))
plt.close()

# ========== SHAP Interaction Significance Analysis (NEW) ==========
# 获取 SHAP 交互作用值
shap_interaction_values = explainer.shap_interaction_values(X_test)

# 计算交互作用显著性
interaction_significance = {}

# 获取特征重要性排序前 9 名的特征
feature_importances = model.feature_importances_
top_9_features_idx = np.argsort(feature_importances)[::-1][:9]
top_9_features = [cols[i] for i in top_9_features_idx]

print(f"\nAnalyzing SHAP interactions for top 9 features: {', '.join(top_9_features)}")

for i in range(len(top_9_features)):
    for j in range(i, len(top_9_features)):  # 从i开始，避免重复计算 (A,B) 和 (B,A)
        feature_i_name = top_9_features[i]
        feature_j_name = top_9_features[j]

        # 获取这两个特征的SHAP主效应值
        shap_vals_i = shap_values.values[:, top_9_features_idx[i]]
        shap_vals_j = shap_values.values[:, top_9_features_idx[j]]

        # 计算每对特征 SHAP 主效应之间的皮尔逊相关系数
        corr, p_value = pearsonr(shap_vals_i, shap_vals_j)

        # 存储相关系数和p值
        interaction_significance[(feature_i_name, feature_j_name)] = (round(corr, 4), p_value)

# 将交互作用显著性存储为 DataFrame
interaction_df = pd.DataFrame(columns=["Feature", "Other_Feature", "Pearson_Correlation", "P_Value", "Significance"])

# 遍历计算的相关系数和 p-value，添加显著性标记
for (feature_i, feature_j), (corr, p_value) in interaction_significance.items():
    significance = ''
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'

    interaction_df = pd.concat([interaction_df, pd.DataFrame({
        "Feature": [feature_i],
        "Other_Feature": [feature_j],
        "Pearson_Correlation": [corr],
        "P_Value": [p_value],
        "Significance": [significance]
    })], ignore_index=True)

# 确保输出目录存在
os.makedirs('2023', exist_ok=True)

# 保存交互作用显著性分析结果到 CSV
interaction_csv_path = os.path.join('2023', 'shap_interaction_significance.csv')
interaction_df.to_csv(interaction_csv_path, index=False, float_format='%.6f')
print(f'SHAP interaction significance results saved to: {interaction_csv_path}')

# ========== Model Saving ==========
joblib.dump(model, os.path.join(output_dir, 'xgboost_model.pkl'))
print('\nAll results have been saved to the XGB directory')