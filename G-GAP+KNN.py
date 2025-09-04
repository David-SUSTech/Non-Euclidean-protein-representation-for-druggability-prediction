import re
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import itertools
import warnings
import random

# --- 全局随机种子设置 ---
# 设置一个固定的随机种子以确保结果的可复现性
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# -------------------------

warnings.filterwarnings('ignore')


class PeptideDataset:
    def __init__(self, ids_file='No_redundancy_ids.txt', seq_file='氨基酸序列（融合用）.txt'):
        try:
            # 兼容旧的文件名
            if ids_file == 'common_protein_ids.txt':
                ids_file = 'No_redundancy_ids.txt'
            with open(ids_file, 'r', encoding='utf-8') as f:
                common_ids_content = f.read()

            with open(seq_file, 'r', encoding='utf-8') as f:
                sequence_content = f.read()
        except FileNotFoundError as e:
            print(f"错误：找不到文件 {e.filename}。请确保文件与脚本在同一目录下。")
            exit()

        self.sequences = []
        self.labels = []
        self.ids = []

        self.common_protein_ids = self._read_common_protein_ids(common_ids_content)
        self._parse_and_filter_file(sequence_content)

        if not self.sequences:
            print("错误：未能从文件中加载任何序列数据。请检查文件内容和格式。")
            exit()

    def _read_common_protein_ids(self, content):
        common_ids = {line.strip() for line in content.split('\n') if line.strip()}
        print(f"从ID文件加载了 {len(common_ids)} 个唯一的公共ID。")
        return common_ids

    def _parse_and_filter_file(self, content):
        pattern = r"(\w+)'(\w+)'\s*:\s*\(Label:\s*(\d+)\)\s*([A-Z]+)"
        matches = re.findall(pattern, content)

        for match in matches:
            protein_id, sub_id, label, sequence = match
            full_id = f"{protein_id}'{sub_id}'"

            if full_id in self.common_protein_ids:
                self.sequences.append(sequence)
                self.labels.append(int(label))
                self.ids.append(full_id)

        print(f"匹配公共ID后，共加载了 {len(self.sequences)} 条序列。")

    def get_dataframe(self):
        df = pd.DataFrame({
            'ID': self.ids,
            'Sequence': self.sequences,
            'Label': self.labels
        })
        return df


class GGAPFeatureExtractor:
    """
    G-gap Dipeptide Composition Feature Extractor.
    为序列生成G-gap二肽组成特征。
    """

    def __init__(self, lambda_param=1):
        """
        初始化特征提取器。

        Args:
            lambda_param (int): The maximum gap value (g). The gaps will range from 0 to lambda_param-1.
                                 例如, lambda_param=1 表示只计算 gap=0 的情况。
                                 lambda_param=2 表示计算 gap=0 和 gap=1 的情况。
        """
        self.lambda_param = lambda_param
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.dipeptides = [aa1 + aa2 for aa1 in self.amino_acids for aa2 in self.amino_acids]
        print(f"GGAP 特征提取器已初始化，lambda_param = {self.lambda_param}。特征维度将是 {self.lambda_param * 400}。")

    def _count_dipeptides_with_gap(self, sequence, gap):
        """
        计算给定序列和间隔的二肽计数。
        """
        dipeptide_counts = {dp: 0 for dp in self.dipeptides}
        seq_len = len(sequence)

        if seq_len <= gap + 1:
            return dipeptide_counts

        for i in range(seq_len - gap - 1):
            dipeptide = sequence[i] + sequence[i + gap + 1]
            if dipeptide in dipeptide_counts:
                dipeptide_counts[dipeptide] += 1

        return dipeptide_counts

    def extract_features(self, sequences):
        """
        从序列列表中提取G-gap特征。
        """
        all_features = []
        for seq in sequences:
            sequence_feature_vector = []
            for g in range(self.lambda_param):
                dipeptide_counts = self._count_dipeptides_with_gap(seq, g)
                counts_list = list(dipeptide_counts.values())
                total_counts = sum(counts_list)

                if total_counts > 0:
                    gap_feature = [count / total_counts for count in counts_list]
                else:
                    gap_feature = [0.0] * 400

                sequence_feature_vector.extend(gap_feature)

            all_features.append(np.array(sequence_feature_vector))

        return np.array(all_features)


class ModelEvaluator:
    @staticmethod
    def perform_loocv_evaluation(classifier, X, y):
        print("\n===== 留一交叉验证（LOOCV）性能评估 =====")
        y_pred_all = []
        y_true_all = []
        loo = LeaveOneOut()

        y_np = np.array(y)

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            y_pred_all.extend(y_pred)
            y_true_all.extend(y_test)

        print("\n--- LOOCV 最终评估报告 ---")
        print(f"准确率 (Accuracy): {accuracy_score(y_true_all, y_pred_all):.4f}")
        print(f"精确率 (Precision): {precision_score(y_true_all, y_pred_all):.4f}")
        print(f"召回率 (Recall): {recall_score(y_true_all, y_pred_all):.4f}")
        print(f"F1得分 (F1 Score): {f1_score(y_true_all, y_pred_all):.4f}")
        print("\n详细分类报告:")
        print(classification_report(y_true_all, y_pred_all, target_names=['Class 0', 'Class 1']))

        return {
            'accuracy': accuracy_score(y_true_all, y_pred_all),
            'precision': precision_score(y_true_all, y_pred_all),
            'recall': recall_score(y_true_all, y_pred_all),
            'f1_score': f1_score(y_true_all, y_pred_all)
        }


class ProteinClassifier:
    def __init__(self, random_state=42):
        self.feature_extractor = GGAPFeatureExtractor(lambda_param=1)
        self.scaler = StandardScaler()
        self.evaluator = ModelEvaluator()
        self.optimal_params = None
        self.random_state = random_state

    def find_optimal_knn_params_loocv(self, X_scaled, y, param_grid):
        """
        使用LOOCV和网格搜索寻找KNN的最佳超参数
        """
        print(f"\n===== 开始通过LOOCV寻找KNN最佳超参数 =====")
        best_score = -1
        best_params = {}
        y_np = np.array(y)

        keys = param_grid.keys()
        values = param_grid.values()
        all_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for params in all_param_combinations:
            knn = KNeighborsClassifier(**params, n_jobs=-1)

            try:
                scores = cross_val_score(knn, X_scaled, y_np, cv=LeaveOneOut(), scoring='accuracy')
                mean_score = scores.mean()
                print(f"测试参数: {params}, LOOCV 平均准确率: {mean_score:.4f}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            except Exception as e:
                print(f"参数组合 {params} 遇到错误: {e}")
                continue

        print(f"\n--- 搜索完成 ---")
        print(f"找到的最佳参数组合: {best_params} (准确率: {best_score:.4f})")
        return best_params

    def train_and_evaluate(self, X, y):
        # 1. 特征提取
        print("\n===== 步骤1：使用 G-gap 进行特征提取 =====")
        X_features = self.feature_extractor.extract_features(X)
        print(f"特征提取完成，特征矩阵维度: {X_features.shape}")

        # 2. 数据标准化
        print("\n===== 步骤2：数据标准化 =====")
        X_scaled = self.scaler.fit_transform(X_features)
        y_np = np.array(y)
        print("数据标准化完成。")

        # 3. 定义KNN参数网格并通过LOOCV寻找最佳参数
        param_grid_for_search = {
            'n_neighbors': [5, 7, 9, 11],
            'weights': ['uniform'],
            'metric': ['manhattan', 'euclidean']
        }

        self.optimal_params = self.find_optimal_knn_params_loocv(X_scaled, y_np, param_grid_for_search)

        if not self.optimal_params:
            print("错误：未能找到最佳KNN参数，终止评估。")
            return

        # 4. 使用最佳参数创建最终模型，并用LOOCV进行评估
        print(f"\n===== 步骤4：使用最佳参数 {self.optimal_params} 进行最终评估 =====")
        final_classifier = KNeighborsClassifier(
            **self.optimal_params,
            n_jobs=-1
        )

        cv_results = self.evaluator.perform_loocv_evaluation(
            final_classifier, X_scaled, y_np
        )

        return final_classifier, cv_results


def main():
    print("--- 开始数据加载与预处理 ---")
    try:
        dataset = PeptideDataset(ids_file='No_redundancy_ids.txt')
    except FileNotFoundError:
        print("未找到 'No_redundancy_ids.txt'，尝试使用 'common_protein_ids.txt'...")
        try:
            dataset = PeptideDataset(ids_file='common_protein_ids.txt')
        except FileNotFoundError:
            print("错误: 'common_protein_ids.txt' 也未找到。请确保ID文件存在。")
            return

    print("\n原始数据预览:")
    print(dataset.get_dataframe().head())

    X = dataset.sequences
    y = dataset.labels

    print("\n--- 开始模型训练与评估 ---")
    # 传递在脚本顶部设置的全局随机种子
    classifier = ProteinClassifier(random_state=SEED)
    classifier.train_and_evaluate(X, y)
    print("\n--- 全流程执行完毕 ---")


if __name__ == '__main__':
    main()
