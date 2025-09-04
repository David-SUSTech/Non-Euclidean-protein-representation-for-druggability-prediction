import pandas as pd
import numpy as np
import re
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# --- 全局设置 ---
# 文件列表和对应的模型名称
FILES_TO_PROCESS = [
    ("No_redundancy_dtw.csv", "DTW"),
    ("No_redundancy_sec.csv", "二级结构"),
    ("No_redundancy_hyd.csv", "疏水性"),
    ("No_redundancy_density.csv", "密度")
]
# 候选的K值列表
K_CANDIDATES = [5, 7, 9, 11]


# --- 辅助函数 ---
def extract_label(name):
    """从索引名称中安全地提取标签。"""
    match = re.search(r'Label: (\d+)', str(name))
    if match:
        return int(match.group(1))
    raise ValueError(f'标签无法从 "{name}" 中提取')


def majority_vote(predictions):
    """
    对一组预测结果进行多数投票。
    例如: predictions = [1, 2, 1, 3] -> 返回 1
    如果出现平票，默认返回第一个出现的最多的候选项。
    """
    vote_counts = Counter(predictions)
    winner = vote_counts.most_common(1)[0][0]
    return winner


def calculate_metrics(y_true, y_pred):
    """计算四个评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return accuracy, precision, recall, f1


# --- 数据加载 ---
print("===== 正在加载所有数据矩阵... =====")
all_data = {}
y_labels = None
try:
    for file, model_name in FILES_TO_PROCESS:
        df = pd.read_csv(file, index_col=0)
        all_data[model_name] = df.values
        if y_labels is None:
            # 仅需从第一个文件提取一次标签即可
            y_labels = np.array([extract_label(name) for name in df.index])
    print("所有数据文件加载成功。")
except FileNotFoundError as e:
    print(f"错误: 文件未找到 ({e.filename})，程序终止。")
    exit()
except Exception as e:
    print(f"加载数据时发生错误: {e}")
    exit()

# --- 阶段一: 为每个独立模型寻找最优K值 (使用LOOCV) ---
print("\n===== 阶段一: 为每个独立模型寻找最优K值 (使用LOOCV) =====")
best_k_values = {}

for model_name, X_matrix in all_data.items():
    print(f"\n--- 正在为模型 '{model_name}' 寻找最优K值 ---")

    best_k_for_model = 0
    highest_accuracy = 0

    for k in K_CANDIDATES:
        if k >= len(y_labels):
            print(f"K值 {k} 大于或等于样本数，停止搜索。")
            break

        knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        y_pred = []
        loo = LeaveOneOut()

        for train_idx, test_idx in loo.split(X_matrix):
            X_train_fold, y_train_fold = X_matrix[np.ix_(train_idx, train_idx)], y_labels[train_idx]
            X_test_fold = X_matrix[np.ix_(test_idx, train_idx)]

            knn.fit(X_train_fold, y_train_fold)
            pred = knn.predict(X_test_fold)
            y_pred.append(pred[0])

        accuracy = accuracy_score(y_labels, y_pred)
        print(f"  - 测试 K={k}, 准确率: {accuracy:.4f}")

        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_k_for_model = k

    best_k_values[model_name] = best_k_for_model
    print(f"  > 模型 '{model_name}' 的最优K值为: {best_k_for_model} (准确率: {highest_accuracy:.4f})")

# --- 阶段二: 使用最优K值进行多数投票集成 (使用LOOCV) ---
print("\n===== 阶段二: 执行多数投票集成 (使用LOOCV) =====")
print("将使用各模型的最优K值进行集成评估...")

# 初始化每个模型的分类器
classifiers = {
    model_name: KNeighborsClassifier(n_neighbors=k, metric='precomputed')
    for model_name, k in best_k_values.items()
}

# 存储集成模型的最终预测结果
ensemble_predictions = []
true_labels = []

# 开始LOOCV
loo = LeaveOneOut()
for train_idx, test_idx in loo.split(y_labels):
    # 当前测试样本的真实标签
    y_test_sample = y_labels[test_idx]
    true_labels.append(y_test_sample[0])

    # 存储当前测试样本来自四个模型的预测结果
    current_sample_preds = []

    # 遍历每一个模型
    for model_name, knn in classifiers.items():
        # 获取对应模型的数据矩阵
        X_matrix = all_data[model_name]

        # 分割训练和测试数据
        X_train_fold = X_matrix[np.ix_(train_idx, train_idx)]
        y_train_fold = y_labels[train_idx]
        X_test_fold = X_matrix[np.ix_(test_idx, train_idx)]

        # 训练并预测
        knn.fit(X_train_fold, y_train_fold)
        pred = knn.predict(X_test_fold)
        current_sample_preds.append(pred[0])

    # 对当前样本的四个预测结果进行多数投票
    final_pred = majority_vote(current_sample_preds)
    ensemble_predictions.append(final_pred)

# --- 最终结果评估 ---
# 计算集成模型的四个评估指标
ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_f1 = calculate_metrics(true_labels,
                                                                                        ensemble_predictions)

print("\n===== 最终评估结果 =====")
print("各模型的独立最优表现：")
for model_name, k in best_k_values.items():
    # 重新计算该模型的四个指标
    knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
    y_p = []
    for tr_i, te_i in LeaveOneOut().split(all_data[model_name]):
        knn.fit(all_data[model_name][np.ix_(tr_i, tr_i)], y_labels[tr_i])
        y_p.append(knn.predict(all_data[model_name][np.ix_(te_i, tr_i)])[0])

    acc, prec, rec, f1 = calculate_metrics(y_labels, y_p)
    print(f"  - {model_name:<10} (K={k}): 准确率={acc:.4f}, 精确率={prec:.4f}, 召回率={rec:.4f}, F1分数={f1:.4f}")

print("\n集成模型表现：")
print(f"  - 多数投票集成模型:")
print(f"    · 准确率 (Accuracy): {ensemble_accuracy:.4f}")
print(f"    · 精确率 (Precision): {ensemble_precision:.4f}")
print(f"    · 召回率 (Recall): {ensemble_recall:.4f}")
print(f"    · F1分数 (F1-Score): {ensemble_f1:.4f}")