import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             mean_absolute_percentage_error, accuracy_score,
                             classification_report, confusion_matrix)
import joblib  # 用于保存/加载模型

# ==========================================
# 绘图基础设置 (更稳健的字体配置)
# ==========================================
import platform
sys_name = platform.system()
if sys_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
elif sys_name == "Darwin":  # Mac
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据加载与医学特征工程
# ==========================================
print(">>> 1. 正在加载数据并构造衍生医学特征...")
try:
    df = pd.read_csv('final_dataset_top14_with_age.csv', encoding='gbk')
    print(f"✅ 数据加载成功，共 {df.shape[0]} 行，{df.shape[1]} 列")
    print("📋 数据集列名：", df.columns.tolist())  # 打印列名，方便核对
except FileNotFoundError:
    print("❌ 未找到文件，请确保 'final_dataset_top14_with_age.csv' 在当前目录")
    exit()

# --------------------------
# 关键修改1：安全的特征工程 (先检查列是否存在)
# --------------------------
required_cols = ['甘油三酯', '高密度脂蛋白胆固醇', '*天门冬氨酸氨基转换酶', '*丙氨酸氨基转换酶', '血糖']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"❌ 数据集缺少必要列：{missing_cols}，请检查数据！")
    exit()

# 构造医学衍生特征 (防除0报错)
df['TG_HDL_Ratio'] = df['甘油三酯'] / df['高密度脂蛋白胆固醇'].replace(0, np.nan)
df['AST_ALT_Ratio'] = df['*天门冬氨酸氨基转换酶'] / df['*丙氨酸氨基转换酶'].replace(0, np.nan)

# 填补缺失值
df = df.fillna(df.median())

X = df.drop(columns=['血糖'])
y = df['血糖']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ 数据集划分完成：训练集 {X_train.shape[0]} 例，测试集 {X_test.shape[0]} 例")

# ==============================================================================
# ================= 优化方案一：增强型回归模型 (Log变换平滑异常值) =================
# ==============================================================================
print("\n>>> 2. [方案一] 正在训练增强型回归模型 (采用 np.log1p 目标变换)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 对目标变量进行对数变换
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

def objective_reg(trial):
    param = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 80, 400),  # 稍微扩大搜索范围
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1.0),  # 新增：防止过拟合的正则化参数
        'random_state': 42
    }
    model = xgb.XGBRegressor(**param)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_log, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

study_reg = optuna.create_study(direction='minimize')
print("⏳ 正在进行Optuna超参数搜索 (30 trials)...")
study_reg.optimize(objective_reg, n_trials=100)

best_model_reg = xgb.XGBRegressor(**study_reg.best_params, objective='reg:squarederror', random_state=42)
best_model_reg.fit(X_train, y_train_log)

# 预测并逆变换
y_pred_log = best_model_reg.predict(X_test)
y_pred_reg = np.expm1(y_pred_log)

# 评估回归模型
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
mae = mean_absolute_error(y_test, y_pred_reg)
# 安全计算MAPE (防止真实值为0导致无穷大)
non_zero_mask = y_test != 0
mape = mean_absolute_percentage_error(y_test[non_zero_mask], y_pred_reg[non_zero_mask]) if non_zero_mask.sum() > 0 else np.nan
r2 = r2_score(y_test, y_pred_reg)

print("=== 增强型回归模型评价指标 ===")
print(f"RMSE : {rmse:.4f} mmol/L")
print(f"MAE  : {mae:.4f} mmol/L")
print(f"MAPE : {mape*100:.2f}%" if not np.isnan(mape) else "MAPE : (因真实值含0无法计算)")
print(f"R²   : {r2:.4f}")

# --------------------------
# 新增：保存回归模型
# --------------------------
joblib.dump(best_model_reg, 'best_regression_model.pkl')
print("✅ 回归模型已保存为 'best_regression_model.pkl'")

# ==============================================================================
# ================= 优化方案二：临床风险分层模型 (回归转分类) ===================
# ==============================================================================
print("\n>>> 3. [方案二] 正在训练临床风险分层模型 (分类器)...")

# 定义分类规则
def categorize_glucose(val):
    if val < 6.1: return 0
    elif 6.1 <= val < 7.0: return 1
    else: return 2

y_train_class = y_train.apply(categorize_glucose)
y_test_class = y_test.apply(categorize_glucose)

# --------------------------
# 关键修改2：打印类别分布，确认样本不平衡
# --------------------------
print("📊 训练集类别分布：")
print(y_train_class.value_counts().sort_index().rename({0:'正常', 1:'血糖受损', 2:'糖尿病'}))

# --------------------------
# 关键修改3：计算样本权重，解决类别不平衡
# --------------------------
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_class)

def objective_cls(trial):
    param = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'n_estimators': trial.suggest_int('n_estimators', 80, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'random_state': 42
    }
    model = xgb.XGBClassifier(**param)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_class, test_size=0.2, random_state=42)
    # 传入样本权重
    tr_weights = compute_sample_weight(class_weight='balanced', y=y_tr)
    model.fit(X_tr, y_tr, sample_weight=tr_weights, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study_cls = optuna.create_study(direction='maximize')
print("⏳ 正在进行Optuna超参数搜索 (30 trials)...")
study_cls.optimize(objective_cls, n_trials=30)

best_model_cls = xgb.XGBClassifier(**study_cls.best_params, objective='multi:softmax', num_class=3, random_state=42)
# 训练时传入样本权重
best_model_cls.fit(X_train, y_train_class, sample_weight=sample_weights)
y_pred_cls = best_model_cls.predict(X_test)

print("\n=== 临床风险分层模型评估 (分类) ===")
print(f"总体准确率 (Accuracy): {accuracy_score(y_test_class, y_pred_cls):.4f}")
print("\n分类详细报告:")
# --------------------------
# 关键修改4：添加 zero_division=0，彻底消除警告
# --------------------------
print(classification_report(y_test_class, y_pred_cls,
                            target_names=['正常(<6.1)', '血糖受损(6.1-7.0)', '糖尿病(>=7.0)'],
                            zero_division=0))

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_class, y_pred_cls)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['正常', '血糖受损', '糖尿病'],
            yticklabels=['正常', '血糖受损', '糖尿病'])
plt.title('临床风险分层混淆矩阵\n(纵轴:真实情况 | 横轴:预测情况)', fontsize=15)
plt.ylabel('真实分类', fontsize=12)
plt.xlabel('模型预测分类', fontsize=12)
plt.tight_layout()
plt.savefig('分类混淆矩阵.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 分类混淆矩阵已保存为 '分类混淆矩阵.png'")

# --------------------------
# 新增：保存分类模型
# --------------------------
# joblib.dump(best_model_cls, 'best_classification_model.pkl')
# print("✅ 分类模型已保存为 'best_classification_model.pkl'")
# print("\n🎉 所有任务执行完毕！")