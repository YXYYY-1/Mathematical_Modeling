import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ==========================================
# 绘图基础设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows请用SimHei，Mac请改为['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 第一部分：数据清洗与变量初步筛选
# ==========================================
print(">>> 正在进行数据清洗...")
df = pd.read_csv('附件1：有血糖值的检测数据.csv', encoding='gbk')
df = df.drop(columns=['id', '体检日期'], errors='ignore')

# 处理性别
df['性别'] = df['性别'].map({'男': 1, '女': 0})
df = df.dropna(subset=['性别'])

# 删除高缺失率指标
missing_ratio = df.isnull().mean()
df = df.drop(columns=missing_ratio[missing_ratio > 0.5].index)

# 强制转换数值并填补众数
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 1%-99% 科学缩尾处理 (Winsorization)
print(">>> 正在执行 1%-99% 的科学缩尾处理 (Winsorization)...")
for col in df.columns:
    if col not in ['性别', '血糖']:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])

# 剔除低相关性变量 (< 0.05)
correlations = df.corr()['血糖'].abs()
low_corr_cols = correlations[correlations < 0.05].index
df_filtered = df.drop(columns=[col for col in low_corr_cols if col != '血糖'])

# ==========================================
# 第二部分：★★★ VIF 多重共线性检验与剔除 ★★★
# ==========================================
print("\n>>> 正在执行 VIF 多重共线性检验 (迭代剔除 VIF > 10 的变量)...")

# 提取自变量 X (排除因变量血糖)
X = df_filtered.drop(columns=['血糖'])


def calculate_vif_and_drop(X_df, threshold=10.0):
    """
    迭代计算 VIF，每次剔除 VIF 最高且大于阈值的变量
    """
    variables = list(X_df.columns)

    while True:
        # 【极其重要】：计算VIF必须添加常数项，否则非截距模型的VIF会被严重高估
        X_with_const = add_constant(X_df[variables])

        # 计算每个变量的 VIF
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                           for i in range(X_with_const.shape[1])]

        # 排除常数项本身
        vif_data = vif_data[vif_data['Feature'] != 'const']

        max_vif = vif_data['VIF'].max()
        max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']

        if max_vif > threshold:
            print(f"剔除变量: '{max_vif_feature}' (VIF = {max_vif:.2f} > {threshold})")
            variables.remove(max_vif_feature)
        else:
            print("\n所有剩余变量的 VIF 均小于 10，共线性问题已解决。")
            print(vif_data.sort_values(by='VIF', ascending=False).to_string(index=False))
            break

    return variables


# 执行 VIF 筛选
final_features = calculate_vif_and_drop(X, threshold=10.0)

# 组合最终数据集并保存
final_columns = final_features + ['血糖']
df_final = df_filtered[final_columns]
df_final.to_csv('cleaned_data.csv', encoding='gbk', index=False)
print("\n>>> VIF处理完成！已输出：cleaned_datas.csv")

# ==========================================
# 第三部分：绘制经过 VIF 筛选后的 Pearson 相关系数柱状图
# ==========================================
print("\n>>> 正在绘制最终变量的 Pearson 相关系数柱状图...")

final_correlations = df_final.corr()['血糖'].abs().drop('血糖').sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))  # 动态调整画布高度
bars = ax.barh(final_correlations.index, final_correlations.values, color='#4C72B0', edgecolor='black', height=0.7)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{width:.3f}', va='center', ha='left', fontsize=10, color='black')

ax.set_title('Pearson 相关系数', fontsize=16, pad=15)
ax.set_xlabel('Pearson 相关系数', fontsize=12)
ax.set_ylabel('体检指标', fontsize=12)

ax.xaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Pearson相关系数柱状图.png', dpi=300, bbox_inches='tight')
plt.show()
print("绘图完成！已保存为图片：Pearson相关系数柱状图.png")