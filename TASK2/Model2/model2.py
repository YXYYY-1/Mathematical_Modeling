# 安装依赖（首次运行时执行，之后可注释）
# !pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn joblib

# -------------------------- 1. 依赖库导入 --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# 解决中文显示问题（适配Windows/macOS）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 2. 数据预处理（含异常值处理） --------------------------
def load_and_clean_data(data_path, target_col="血糖"):
    """
    功能：加载数据并清洗（处理缺失值、异常值、区分特征类型）
    参数：data_path-数据文件路径（CSV/Excel），target_col-血糖值列名
    返回：特征X、目标变量y、分类特征列表、数值特征列表
    """
    # 加载数据
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path, encoding='gbk')
    else:  # 支持Excel格式
        df = pd.read_excel(data_path)
    print(f"原始数据形状：{df.shape}")

    # 👇 新增：打印所有列名，帮你核对！👇
    print("\n=== 数据所有列名（请核对TARGET_COL是否存在） ===")
    print(df.columns.tolist())
    print("==============================================\n")

    # 检查目标列是否存在
    if target_col not in df.columns:
        raise ValueError(f"错误：目标列'{target_col}'不存在！请检查列名，或修改TARGET_COL为上述列表中的一个。")

    # 区分分类特征与数值特征
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_features:
        num_features.remove(target_col)  # 剔除目标变量，避免混入特征

    # 处理缺失值（分类特征用众数，数值特征用中位数，抗极端值）
    for col in cat_features:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in num_features:
        df[col].fillna(df[col].median(), inplace=True)

    # 处理异常值（IQR方法，医疗数据常用，避免删除正常极端值）
    for col in num_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)  # 用边界值替换异常值
    print(f"清洗后数据形状：{df.shape}")

    # 分离特征与目标变量
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y, cat_features, num_features

# -------------------------- 3. 特征优化（筛选+交互特征） --------------------------
def optimize_features(X, y, cat_features, num_features, k=10):
    """
    功能：优化特征（分类特征编码、构建交互特征、筛选高相关特征）
    参数：X-原始特征，y-目标变量，cat_features-分类特征列表，num_features-数值特征列表，k-保留Top K特征
    返回：优化后特征X_optimized、筛选后的特征名列表
    """
    # 1. 分类特征One-Hot编码（避免类别顺序影响，drop_first防多重共线性）
    X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=True)
    print(f"分类特征编码后数据形状：{X_encoded.shape}")

    # 2. 构建数值特征交互项（捕捉协同影响，如BMI×年龄）
    if len(num_features) >= 2:
        # 选择Top3重要数值特征构建交互项（避免特征爆炸）
        top_num_idx = SelectKBest(f_regression, k=min(3, len(num_features))).fit(X[num_features], y).get_support(indices=True)
        top_num_features = [num_features[i] for i in top_num_idx]
        # 两两构建交互项
        for i in range(len(top_num_features)):
            for j in range(i+1, len(top_num_features)):
                interact_col = f"{top_num_features[i]}_×_{top_num_features[j]}"
                X_encoded[interact_col] = X[top_num_features[i]] * X[top_num_features[j]]
    print(f"构建交互特征后数据形状：{X_encoded.shape}")

    # 3. 筛选Top K高相关特征（F检验，保留对血糖影响显著的特征）
    selector = SelectKBest(f_regression, k=min(k, X_encoded.shape[1]))
    X_optimized = selector.fit_transform(X_encoded, y)
    selected_cols = X_encoded.columns[selector.get_support()]  # 筛选后的特征名
    print(f"最终保留的特征：{list(selected_cols)}")

    return X_optimized, selected_cols

# -------------------------- 4. 多模型训练与调优 --------------------------
def train_optimized_models(X_train, y_train, X_test, y_test):
    """
    功能：训练3种Boosting模型（Gradient Boosting/LightGBM/CatBoost）并随机搜索调优
    参数：训练集(X_train,y_train)、测试集(X_test,y_test)
    返回：最优模型字典、模型结果DataFrame、特征标准化器
    """
    # 特征标准化（RobustScaler抗异常值，比StandardScaler更适合医疗数据）
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 模型配置（含超参数搜索范围）
    models_config = {
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "param_dist": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
                "min_samples_split": [2, 5, 10]
            }
        },
        "LightGBM": {
            "model": LGBMRegressor(random_state=42, verbose=-1),  # verbose=-1隐藏训练日志
            "param_dist": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            }
        },
        "CatBoost": {
            "model": CatBoostRegressor(random_state=42, verbose=0),  # verbose=0隐藏训练日志
            "param_dist": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
                "l2_leaf_reg": [1, 3, 5]
            }
        }
    }

    # 随机搜索调优（5折交叉验证，采样10组参数，平衡效果与速度）
    best_models = {}
    results_list = []
    for model_name, config in models_config.items():
        print(f"\n正在调优 {model_name} 模型...")
        random_search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["param_dist"],
            n_iter=10,  # 采样10组参数
            cv=5,
            scoring="r2",  # 以R²为核心评估指标
            n_jobs=-1,  # 利用所有CPU核心加速
            random_state=42
        )
        random_search.fit(X_train_scaled, y_train)
        best_model = random_search.best_estimator_
        best_models[model_name] = best_model

        # 计算模型评估指标
        y_pred = best_model.predict(X_test_scaled)
        r2 = round(r2_score(y_test, y_pred), 4)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        mae = round(mean_absolute_error(y_test, y_pred), 4)
        mape = round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2)

        # 保存结果
        results_list.append({
            "模型名称": model_name,
            "最优参数": str(random_search.best_params_),  # 转字符串便于显示
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE(%)": mape
        })
        print(f"{model_name} 调优完成：R²={r2}，RMSE={rmse}")

    # 整理结果为DataFrame
    results_df = pd.DataFrame(results_list)
    return best_models, results_df, scaler

# -------------------------- 5. 结果可视化 --------------------------
def plot_model_results(best_models, X_test, y_test, scaler, selected_cols):
    """
    功能：可视化最优模型结果（真实值vs预测值、残差分布、特征重要性）
    参数：最优模型字典、测试集、标准化器、筛选后的特征名
    返回：最优模型名称
    """
    # 选择R²最高的模型作为最优模型
    model_r2_dict = {
        name: r2_score(y_test, model.predict(scaler.transform(X_test)))
        for name, model in best_models.items()
    }
    best_model_name = max(model_r2_dict, key=model_r2_dict.get)
    best_model = best_models[best_model_name]
    y_pred = best_model.predict(scaler.transform(X_test))

    # 绘制1：真实值vs预测值对比图
    plt.figure(figsize=(12, 4))
    # 子图1：散点图
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='#2E86AB', s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'{best_model_name} 真实值vs预测值\n(R²={model_r2_dict[best_model_name]:.4f})', fontsize=12)
    plt.xlabel('真实血糖值 (mmol/L)')
    plt.ylabel('预测血糖值 (mmol/L)')
    plt.grid(alpha=0.3)

    # 子图2：残差分布
    residuals = y_test - y_pred
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, color='#A23B72', bins=15, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.title(f'{best_model_name} 残差分布\n(均值={residuals.mean():.4f})', fontsize=12)
    plt.xlabel('残差 (真实值-预测值)')
    plt.ylabel('频次')
    plt.tight_layout()
    plt.savefig('血糖预测结果对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制2：特征重要性（仅支持有feature_importances_属性的模型）
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        # 按重要性排序
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_cols = [selected_cols[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        # 绘制Top10特征
        sns.barplot(x=sorted_importances[:10], y=sorted_cols[:10], palette='viridis', edgecolor='black')
        plt.title(f'{best_model_name} Top10特征重要性', fontsize=14)
        plt.xlabel('特征重要性')
        plt.tight_layout()
        plt.savefig('特征重要性.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n可视化文件已保存：血糖预测结果对比.png、特征重要性.png")
    return best_model_name

# -------------------------- 6. 主函数（整合全流程） --------------------------
if __name__ == "__main__":
    # -------------------------- 用户需修改的参数（仅2处） --------------------------
    DATA_PATH = "final_dataset_top14_with_age.csv"  # 你的数据文件路径（CSV/Excel）
    # 👇 这里必须改成你CSV文件里真实的血糖列名！👇
    # 比如："空腹血糖"、"血糖"、"餐后血糖"，根据打印的列名修改
    TARGET_COL = "血糖"
    # --------------------------------------------------------------------------------

    # 步骤1：数据加载与清洗
    X, y, cat_features, num_features = load_and_clean_data(DATA_PATH, TARGET_COL)

    # 步骤2：特征优化
    X_optimized, selected_cols = optimize_features(X, y, cat_features, num_features, k=10)  # k=10保留Top10特征

    # 步骤3：划分训练集/测试集（8:2，固定随机种子确保结果可复现）
    X_train, X_test, y_train, y_test = train_test_split(
        X_optimized, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"\n训练集形状：{X_train.shape}，测试集形状：{X_test.shape}")

    # 步骤4：多模型训练与调优
    best_models, results_df, scaler = train_optimized_models(X_train, y_train, X_test, y_test)

    # 步骤5：打印模型结果对比表
    print("\n" + "="*120)
    print("模型优化结果对比表")
    print("="*120)
    print(results_df.to_string(index=False, max_colwidth=50))  # 调整列宽避免参数显示不全

    # 步骤6：可视化最优模型结果
    best_model_name = plot_model_results(best_models, X_test, y_test, scaler, selected_cols)

    # 步骤7：保存最优模型与标准化器（便于后续调用）
    joblib.dump(best_models[best_model_name], f"最优血糖预测模型_{best_model_name}.pkl")
    joblib.dump(scaler, "特征标准化器.pkl")
    print(f"\n模型文件已保存：最优血糖预测模型_{best_model_name}.pkl、特征标准化器.pkl")
    print(f"运行完成！可查看生成的图片和模型文件进行后续分析。")