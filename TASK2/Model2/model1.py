import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据预处理
# 读取数据
df = pd.read_csv("final_dataset_top14_with_age.csv", encoding="gbk")

# 定义特征和目标变量
target_col = '血糖'
X = df.drop(columns=[target_col])
y = df[target_col]

print("=== 数据预处理 ===")
print(f"特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 处理异常值（使用3σ原则去除极端异常值）
mean_val = y.mean()
std_val = y.std()
mask = (y >= mean_val - 3 * std_val) & (y <= mean_val + 3 * std_val)
X_clean = X[mask]
y_clean = y[mask]

print(f"去除异常值前样本数: {len(y)}")
print(f"去除异常值后样本数: {len(y_clean)}")
print(f"去除异常值比例: {(1 - len(y_clean) / len(y)) * 100:.2f}%")

# 数据分割（80%训练集，20%测试集，固定随机种子确保可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 模型定义与优化
# 定义要比较的模型及参数网格
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge Regression': {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1, 5, 10, 20, 50]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 10]
        }
    }
}

# 存储最优模型和结果
best_models = {}
model_performances = []

print("\n=== 模型训练与优化 ===")

for name, config in models.items():
    print(f"\n正在训练 {name}...")

    # 使用GridSearchCV进行参数优化
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )

    # 训练模型
    grid_search.fit(X_train_scaled, y_train)

    # 保存最优模型
    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    # 预测
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    # 计算评价指标
    # 训练集指标
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    # 处理MAPE中可能的零值问题
    train_mape = np.mean(np.abs((y_train - y_pred_train) / np.maximum(np.abs(y_train), 1e-6))) * 100

    # 测试集指标
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    test_mape = np.mean(np.abs((y_test - y_pred_test) / np.maximum(np.abs(y_test), 1e-6))) * 100

    # 存储结果
    model_performances.append({
        'Model': name,
        'Best Params': grid_search.best_params_,
        'Train MAE': train_mae,
        'Train RMSE': train_rmse,
        'Train MAPE(%)': train_mape,
        'Train R²': train_r2,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'Test MAPE(%)': test_mape,
        'Test R²': test_r2
    })

    print(f"{name} 最优参数: {grid_search.best_params_}")
    print(f"{name} 测试集性能: R²={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, MAPE={test_mape:.2f}%")

# 3. 结果整理与可视化
# 创建性能对比表格
performance_df = pd.DataFrame(model_performances)
print("\n=== 所有模型性能对比 ===")
print(performance_df.round(4))

# 找出最优模型（基于测试集R²）
best_model_name = performance_df.loc[performance_df['Test R²'].idxmax(), 'Model']
best_model = best_models[best_model_name]
print(f"\n=== 最优模型 ===")
print(f"模型名称: {best_model_name}")
best_performance = performance_df[performance_df['Model'] == best_model_name].iloc[0]
print(f"测试集 R²: {best_performance['Test R²']:.4f}")
print(f"测试集 RMSE: {best_performance['Test RMSE']:.4f}")
print(f"测试集 MAE: {best_performance['Test MAE']:.4f}")
print(f"测试集 MAPE: {best_performance['Test MAPE(%)']:.2f}%")

# 4. 最优模型可视化
# 预测结果对比图
y_pred_best = best_model.predict(X_test_scaled)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'最优模型: {best_model_name} 性能可视化', fontsize=16, fontweight='bold')

# 1. 实际值vs预测值散点图
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.6, color='#2E86AB', s=30)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='理想预测线')
axes[0, 0].set_xlabel('实际血糖值', fontsize=12)
axes[0, 0].set_ylabel('预测血糖值', fontsize=12)
axes[0, 0].set_title(f'实际值 vs 预测值\nR² = {best_performance["Test R²"]:.4f}', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 2. 残差分布图
residuals = y_test - y_pred_best
axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')
axes[0, 1].set_xlabel('残差（实际值-预测值）', fontsize=12)
axes[0, 1].set_ylabel('频数', fontsize=12)
axes[0, 1].set_title('残差分布', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# 3. 模型性能对比柱状图（R²）
models_list = performance_df['Model'].tolist()
test_r2_list = performance_df['Test R²'].tolist()
colors = ['#F18F01' if model != best_model_name else '#C73E1D' for model in models_list]

axes[1, 0].bar(models_list, test_r2_list, color=colors, alpha=0.8)
axes[1, 0].set_xlabel('模型', fontsize=12)
axes[1, 0].set_ylabel('测试集 R²', fontsize=12)
axes[1, 0].set_title('各模型 R² 对比', fontsize=11)
axes[1, 0].grid(True, alpha=0.3, axis='y')
# 在柱子上添加数值标签
for i, v in enumerate(test_r2_list):
    axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
# 旋转x轴标签
plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

# 4. 各评价指标对比（测试集）
metrics = ['Test RMSE', 'Test MAE', 'Test MAPE(%)']
x_pos = np.arange(len(models_list))
width = 0.25

for i, metric in enumerate(metrics):
    values = performance_df[metric].tolist()
    axes[1, 1].bar(x_pos + i * width, values, width, label=metric, alpha=0.8)

axes[1, 1].set_xlabel('模型', fontsize=12)
axes[1, 1].set_ylabel('指标值', fontsize=12)
axes[1, 1].set_title('各模型测试集指标对比', fontsize=11)
axes[1, 1].set_xticks(x_pos + width)
axes[1, 1].set_xticklabels(models_list)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')
# 旋转x轴标签
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('/mnt/blood_glucose_prediction_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n可视化结果已保存为: blood_glucose_prediction_results.png")

# 5. 特征重要性分析（针对树模型）
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print(f"\n=== {best_model_name} 特征重要性 ===")
    feature_importance = best_model.feature_importances_
    feature_names = X.columns

    # 创建特征重要性DataFrame并排序
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    print(importance_df.round(4))

    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='#2E86AB', alpha=0.8)
    plt.xlabel('重要性得分', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.title(f'{best_model_name} 特征重要性排序', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # 在条形图上添加数值
    for i, v in enumerate(importance_df['Importance'][::-1]):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('/mnt/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存为: feature_importance.png")

# 6. 保存性能结果到CSV
performance_df.to_csv('/mnt/model_performance_results.csv', index=False, encoding='utf-8-sig')
print(f"\n模型性能结果已保存为: model_performance_results.csv")

# 输出最终结论
print(f"\n=== 血糖预测模型构建完成 ===")
print(f"1. 数据概况: {len(X_clean)}个有效样本，{X_clean.shape[1]}个特征")
print(f"2. 最优模型: {best_model_name}")
print(f"3. 关键性能指标（测试集）:")
print(f"   - R²: {best_performance['Test R²']:.4f}")
print(f"   - RMSE: {best_performance['Test RMSE']:.4f}")
print(f"   - MAE: {best_performance['Test MAE']:.4f}")
print(f"   - MAPE: {best_performance['Test MAPE(%)']:.2f}%")
print(f"4. 模型可解释性: 已生成特征重要性分析（针对树模型）")