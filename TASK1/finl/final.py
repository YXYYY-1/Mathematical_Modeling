import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 绘图基础设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False

print(">>> 正在启动数据清洗流水线（保留年龄版）...")

# ==========================================
# 第一步：基础清洗与缩尾处理
# ==========================================
# 1. 读取原始数据
df = pd.read_csv('附件1：有血糖值的检测数据.csv', encoding='gbk')

# 2. 仅删除 id 和 日期，【保留年龄】
df = df.drop(columns=['id', '体检日期'], errors='ignore')

# 3. 处理性别
df['性别'] = df['性别'].map({'男': 1, '女': 0})
df = df.dropna(subset=['性别'])

# 4. 删除高缺失率指标 (>50%)
missing_ratio = df.isnull().mean()
df = df.drop(columns=missing_ratio[missing_ratio > 0.5].index)

# 5. 格式强制转换与众数填补
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 6. 1%-99% 科学缩尾处理
for col in df.columns:
    if col not in ['性别', '血糖']:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

# 7. 剔除低相关性变量 (< 0.05)，但【强制保留年龄】
correlations = df.corr()['血糖'].abs()
low_corr_cols = correlations[correlations < 0.05].index.tolist()
if '年龄' in low_corr_cols:
    low_corr_cols.remove('年龄')  # 特赦年龄
df_filtered = df.drop(columns=[col for col in low_corr_cols if col != '血糖'])

# ==========================================
# 第二步：VIF 多重共线性检验
# ==========================================
print("\n>>> 正在执行 VIF 多重共线性剔除...")
X_vif = df_filtered.drop(columns=['血糖'])


def calculate_vif_and_drop(X_df, threshold=10.0):
    variables = list(X_df.columns)
    while True:
        X_with_const = add_constant(X_df[variables])
        vif_data = pd.DataFrame({
            "Feature": X_with_const.columns,
            "VIF": [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
        })
        vif_data = vif_data[vif_data['Feature'] != 'const']

        max_vif = vif_data['VIF'].max()
        if max_vif > threshold:
            max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
            variables.remove(max_vif_feature)
        else:
            break
    return variables


vif_features = calculate_vif_and_drop(X_vif, threshold=10.0)

# ==========================================
# 第三步：医学先验知识 + LASSO 终极 14 维提取
# ==========================================
print("\n>>> 正在执行 医学特征白名单 与 LASSO 终极降维...")

# 医学四大核心维度的白名单（加入年龄）
medical_features = [
    '年龄', '性别',
    '*丙氨酸氨基转换酶', '*天门冬氨酸氨基转换酶', '*r-谷氨酰基转换酶', '*碱性磷酸酶',
    '尿素', '肌酐', '尿酸',
    '甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇',
    '白细胞计数', '红细胞计数', '血红蛋白', '血小板计数'
]

# 求 VIF剩余特征与医学白名单的交集
available_cols = [col for col in medical_features if col in vif_features]
X_medical = df_filtered[available_cols]
y = df_filtered['血糖']

# 数据标准化（LASSO前必须执行）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_medical)

# 执行 LASSO 交叉验证
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y)

# 提取 LASSO 特征权重绝对值并降序排列
lasso_coefs = pd.Series(np.abs(lasso.coef_), index=X_medical.columns)
lasso_coefs = lasso_coefs.sort_values(ascending=False)

# ★ 核心控量：强制截取排名前 14 的变量 ★
# (剔除权重为 0 的无效变量，并在剩余有效变量中最多取前 14 个)
valid_features = lasso_coefs[lasso_coefs > 0]
target_num = min(14, len(valid_features))
final_top_features = valid_features.head(target_num).index.tolist()

print(f"\n=== 最终入模的 {len(final_top_features)} 个黄金特征 ===")
for i, feat in enumerate(final_top_features):
    print(f"{i + 1}. {feat}")

# 保存最终的 14维 数据集
df_final = df_filtered[final_top_features + ['血糖']]
df_final.to_csv('final_dataset_top14_with_age.csv', encoding='gbk', index=False)
print("\n>>> 降维完成！已输出：final_dataset_top14_with_age.csv")

# ==========================================
# 可视化：Top 14 特征重要性
# ==========================================
# 获取原始带正负号的权重用于作图
plot_coefs = pd.Series(lasso.coef_, index=X_medical.columns)[final_top_features].sort_values(key=abs, ascending=True)

plt.figure(figsize=(10, 8))
colors = ['#d62728' if val > 0 else '#1f77b4' for val in plot_coefs.values]
bars = plt.barh(plot_coefs.index, plot_coefs.values, color=colors, edgecolor='black', height=0.6)

plt.title(f'定的 {len(final_top_features)} 个核心变量 LASSO 权重\n(红色正相关, 蓝色负相关)', fontsize=15, pad=15)
plt.xlabel('标准化回归系数 (重要性)', fontsize=12)
plt.axvline(0, color='black', linewidth=1.2)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('变量权重图.png', dpi=300)
plt.show()
print(">>> 已保存特征重要性条形图：最终14变量权重图.png")