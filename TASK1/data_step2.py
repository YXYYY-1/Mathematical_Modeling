import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 绘图基础设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows请用SimHei，Mac请用['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 第一步：基于医学专家知识的初步降维
# ==========================================
print(">>> 正在加载经过 VIF 处理的数据集...")
df = pd.read_csv('cleaned_data.csv', encoding='gbk')

# 定义医学核心指标字典（剔除边缘形态学指标）
medical_features = {
    '人口特征': ['性别','年龄'],
    '肝功能特征': ['*丙氨酸氨基转换酶', '*天门冬氨酸氨基转换酶', '*r-谷氨酰基转换酶', '*碱性磷酸酶'],
    '肾功能特征': ['尿素', '肌酐', '尿酸'],
    '血液及脂代谢特征': ['甘油三酯', '总胆固醇', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇',
                '白细胞计数', '红细胞计数', '血红蛋白', '血小板计数']
}

# 将字典展平为列表，并与当前数据集中实际存在的列进行取交集（防止某些列在上一步VIF被删而报错）
core_medical_cols = [col for category in medical_features.values() for col in category]
available_cols = [col for col in core_medical_cols if col in df.columns]

# 构建初步经过医学筛选的特征集
X_medical = df[available_cols]
y = df['血糖']

print(f">>> 经过医学常识过滤，保留了 {len(available_cols)} 个核心医学指标。")

# ==========================================
# 第二步：基于 LASSO (L1 正则化) 的算法降维防过拟合
# ==========================================
print(">>> 正在使用 LassoCV 进行特征收缩与终极降维...")

# LASSO 算法对量纲极其敏感，必须先进行标准化 (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_medical)

# 训练带有 5 折交叉验证的 LASSO 回归模型
# cv=5 意味着模型会自己在内部测试寻找最平衡的 alpha 惩罚力度，最大程度防止过拟合
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y)

# 获取特征的回归系数 (被 LASSO 压缩为 0 的特征将被无情剔除)
lasso_coefs = pd.Series(lasso.coef_, index=X_medical.columns)
final_features = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)

print("\n=== 最终存活入模的黄金特征及其权重 (绝对值越大越重要) ===")
print(final_features.to_string())

# ==========================================
# 第三步：保存终极防过拟合数据集
# ==========================================
# 提取最终特征列表
selected_feature_names = final_features.index.tolist()

# 组合最终数据集并保存
df_final = df[selected_feature_names + ['血糖']]
df_final.to_csv('final_dataset.csv', encoding='gbk', index=False)
print("\n>>> 终极降维完成！已输出：final_dataset_medical_lasso.csv")
print(f">>> 特征维度已从最初的 41 维，精简至最核心的 {len(selected_feature_names)} 维，模型将极其稳定。")

# （可选）可视化 LASSO 系数
plt.figure(figsize=(8, 6))
bars = plt.barh(final_features.index[::-1], final_features.values[::-1], color=np.where(final_features.values[::-1]>0, '#d62728', '#1f77b4'))
plt.title('LASSO 筛选后的核心特征权重 (红色正相关，蓝色负相关)', fontsize=14)
plt.xlabel('标准化后的回归系数 (影响力度)', fontsize=12)
plt.axvline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig('LASSO核心特征权重图.png', dpi=300)
plt.show()