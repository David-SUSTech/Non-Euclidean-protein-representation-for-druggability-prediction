import re
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import random

# --- 全局随机种子设置 ---
# 设置一个固定的随机种子以确保结果的可复现性
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# -------------------------

# 定义氨基酸的7种物理化学性质
amino_acid_properties = {
    'A': {'polarity': -0.5, 'van_der_waals_volume': 52, 'hydrophobicity': 1.8, 'secondary_structure': 0.3,
          'solvent_accessibility': 0.2, 'charge': 0.0, 'polarizability': 0.046},
    'R': {'polarity': 1.0, 'van_der_waals_volume': 109, 'hydrophobicity': -4.5, 'secondary_structure': 0.2,
          'solvent_accessibility': 0.8, 'charge': 1.0, 'polarizability': 0.179},
    'N': {'polarity': 0.2, 'van_der_waals_volume': 75, 'hydrophobicity': -3.5, 'secondary_structure': 0.3,
          'solvent_accessibility': 0.6, 'charge': 0.0, 'polarizability': 0.081},
    'D': {'polarity': 0.3, 'van_der_waals_volume': 68, 'hydrophobicity': -3.5, 'secondary_structure': 0.3,
          'solvent_accessibility': 0.7, 'charge': -1.0, 'polarizability': 0.105},
    'C': {'polarity': 0.1, 'van_der_waals_volume': 67, 'hydrophobicity': 2.5, 'secondary_structure': 0.2,
          'solvent_accessibility': 0.3, 'charge': 0.0, 'polarizability': 0.114},
    'E': {'polarity': 0.3, 'van_der_waals_volume': 84, 'hydrophobicity': -3.5, 'secondary_structure': 0.3,
          'solvent_accessibility': 0.7, 'charge': -1.0, 'polarizability': 0.151},
    'Q': {'polarity': 0.2, 'van_der_waals_volume': 85, 'hydrophobicity': -3.5, 'secondary_structure': 0.3,
          'solvent_accessibility': 0.6, 'charge': 0.0, 'polarizability': 0.180},
    'G': {'polarity': 0.0, 'van_der_waals_volume': 48, 'hydrophobicity': -0.4, 'secondary_structure': 0.5,
          'solvent_accessibility': 0.5, 'charge': 0.0, 'polarizability': 0.000},
    'H': {'polarity': 0.5, 'van_der_waals_volume': 87, 'hydrophobicity': -3.2, 'secondary_structure': 0.3,
          'solvent_accessibility': 0.5, 'charge': 0.5, 'polarizability': 0.165},
    'I': {'polarity': -0.5, 'van_der_waals_volume': 93, 'hydrophobicity': 4.5, 'secondary_structure': 0.4,
          'solvent_accessibility': 0.2, 'charge': 0.0, 'polarizability': 0.122},
    'L': {'polarity': -0.5, 'van_der_waals_volume': 96, 'hydrophobicity': 3.8, 'secondary_structure': 0.4,
          'solvent_accessibility': 0.2, 'charge': 0.0, 'polarizability': 0.122},
    'K': {'polarity': 0.5, 'van_der_waals_volume': 101, 'hydrophobicity': -3.9, 'secondary_structure': 0.3,
          'solvent_accessibility': 0.8, 'charge': 1.0, 'polarizability': 0.220},
    'M': {'polarity': -0.5, 'van_der_waals_volume': 94, 'hydrophobicity': 1.9, 'secondary_structure': 0.3,
          'solvent_accessibility': 0.3, 'charge': 0.0, 'polarizability': 0.125},
    'F': {'polarity': -0.5, 'van_der_waals_volume': 135, 'hydrophobicity': 2.8, 'secondary_structure': 0.4,
          'solvent_accessibility': 0.2, 'charge': 0.0, 'polarizability': 0.219},
    'P': {'polarity': 0.0, 'van_der_waals_volume': 90, 'hydrophobicity': -1.6, 'secondary_structure': 0.4,
          'solvent_accessibility': 0.4, 'charge': 0.0, 'polarizability': 0.059},
    'S': {'polarity': 0.3, 'van_der_waals_volume': 54, 'hydrophobicity': -0.8, 'secondary_structure': 0.4,
          'solvent_accessibility': 0.5, 'charge': 0.0, 'polarizability': 0.062},
    'T': {'polarity': 0.2, 'van_der_waals_volume': 71, 'hydrophobicity': -0.7, 'secondary_structure': 0.4,
          'solvent_accessibility': 0.4, 'charge': 0.0, 'polarizability': 0.108},
    'W': {'polarity': -0.5, 'van_der_waals_volume': 163, 'hydrophobicity': -0.9, 'secondary_structure': 0.4,
          'solvent_accessibility': 0.2, 'charge': 0.0, 'polarizability': 0.409},
    'Y': {'polarity': 0.2, 'van_der_waals_volume': 117, 'hydrophobicity': -1.3, 'secondary_structure': 0.4,
          'solvent_accessibility': 0.4, 'charge': 0.0, 'polarizability': 0.210},
    'V': {'polarity': -0.5, 'van_der_waals_volume': 84, 'hydrophobicity': 4.2, 'secondary_structure': 0.4,
          'solvent_accessibility': 0.2, 'charge': 0.0, 'polarizability': 0.105}
}


class PeptideDataset:
    def __init__(self, id_file, seq_file):
        try:
            with open(id_file, 'r', encoding='utf-8') as f:
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
        print(f"类别分布: {pd.Series(self.labels).value_counts().to_dict()}")

    def get_dataframe(self):
        return pd.DataFrame({'ID': self.ids, 'Sequence': self.sequences, 'Label': self.labels})


class PCPseAACFeatureExtractor:
    def __init__(self, w=0.1, lambda_param=5):
        self.w = w
        self.lambda_param = lambda_param
        self.amino_acid_properties = amino_acid_properties
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.prop_keys = list(next(iter(self.amino_acid_properties.values())).keys())
        self.corr_prop_keys = ['polarity', 'hydrophobicity', 'charge', 'secondary_structure']

    def extract_features(self, sequences):
        return np.array([self._extract_single_sequence(seq) for seq in sequences])

    def _extract_single_sequence(self, sequence):
        """为单条序列提取特征"""
        aa_freq = self._calculate_amino_acid_frequency(sequence)
        phys_features = self._calculate_physicochemical_features(sequence)
        seq_corr = self._calculate_sequence_correlation(sequence)

        w_float = float(self.w)
        return np.concatenate([
            aa_freq,
            phys_features,
            w_float * np.array(seq_corr)
        ])

    def _calculate_amino_acid_frequency(self, sequence):
        seq_len = len(sequence)
        if seq_len == 0:
            return np.zeros(len(self.amino_acids))
        return np.array([sequence.count(aa) / seq_len for aa in self.amino_acids])

    def _calculate_physicochemical_features(self, sequence):
        if not sequence:
            return np.zeros(len(self.prop_keys) * 4)

        feature_vector = []
        sequence_props = [[self.amino_acid_properties.get(aa, {}).get(prop, 0) for aa in sequence] for prop in
                          self.prop_keys]

        for prop_values in sequence_props:
            feature_vector.extend([
                np.mean(prop_values),
                np.std(prop_values),
                np.max(prop_values),
                np.min(prop_values)
            ])
        return np.array(feature_vector)

    def _calculate_sequence_correlation(self, sequence):
        L = len(sequence)
        correlations = []
        for k in range(1, self.lambda_param + 1):
            if L <= k:
                correlations.append(0)
                continue

            correlation_sum = 0
            for i in range(L - k):
                aa1_props = self.amino_acid_properties.get(sequence[i], {})
                aa2_props = self.amino_acid_properties.get(sequence[i + k], {})

                prop_diff_sq_sum = np.sum([
                    (aa1_props.get(prop, 0) - aa2_props.get(prop, 0)) ** 2
                    for prop in self.corr_prop_keys
                ])
                correlation_sum += np.sqrt(prop_diff_sq_sum)

            correlations.append(correlation_sum / (L - k))
        return correlations


class ProteinClassifier:
    def __init__(self):
        self.feature_extractor = PCPseAACFeatureExtractor()
        self.optimal_k = None
        self.pipeline = None

    def find_optimal_k_loocv(self, X_features, y, k_range):
        """
        使用带有Pipeline的LOOCV寻找最佳的k值，避免数据泄漏
        """
        print(f"\n===== [修正流程] 开始通过LOOCV寻找最佳K值（范围：{k_range[0]}-{k_range[-1]}）=====")
        k_scores = []
        y = np.array(y)

        for k in k_range:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski', p=2))
            ])
            scores = cross_val_score(pipeline, X_features, y, cv=LeaveOneOut(), scoring='accuracy')
            mean_score = scores.mean()
            k_scores.append(mean_score)
            print(f"K={k}, LOOCV 平均准确率: {mean_score:.4f}")

        best_k_index = np.argmax(k_scores)
        best_k = k_range[best_k_index]
        best_score = k_scores[best_k_index]

        print("\n--- 搜索完成 ---")
        print(f"最佳K值为: {best_k} (准确率: {best_score:.4f})")
        return best_k

    def perform_loocv_evaluation(self, X_features, y, k):
        """
        使用最佳K值进行最终的LOOCV评估
        """
        print(f"\n===== [修正流程] 使用最佳K值 (K={k}) 进行最终评估 =====")

        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2))
        ])

        y_pred_all = []
        y_true_all = []
        loo = LeaveOneOut()

        # 使用y的numpy数组版本
        y_np = np.array(y)

        for train_index, test_index in loo.split(X_features):
            X_train, X_test = X_features[train_index], X_features[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]

            final_pipeline.fit(X_train, y_train)
            y_pred = final_pipeline.predict(X_test)

            y_pred_all.extend(y_pred)
            y_true_all.extend(y_test)

        print("\n--- LOOCV 最终评估报告 ---")
        print(f"准确率 (Accuracy): {accuracy_score(y_true_all, y_pred_all):.4f}")
        print(f"精确率 (Precision): {precision_score(y_true_all, y_pred_all):.4f}")
        print(f"召回率 (Recall): {recall_score(y_true_all, y_pred_all):.4f}")
        print(f"F1得分 (F1 Score): {f1_score(y_true_all, y_pred_all):.4f}")
        print("\n详细分类报告:")
        print(classification_report(y_true_all, y_pred_all, target_names=['Class 0', 'Class 1']))

        self.pipeline = final_pipeline.fit(X_features, y_np)
        print("\n最终模型已在全部数据上完成训练。")

    def train_and_evaluate(self, X, y):
        print("\n===== 步骤1：特征提取 =====")
        X_features = self.feature_extractor.extract_features(X)
        y_np = np.array(y)

        k_upper_bound = len(y) - 1 if len(y) > 1 else 1
        # 搜索奇数K值，避免平票, 上限设为20防止过拟合
        k_range = list(range(5, min(20, k_upper_bound) + 1, 2))

        if not k_range:
            print("警告：样本数量过少，无法进行K值搜索。将使用默认K=3。")
            self.optimal_k = 3
        else:
            self.optimal_k = self.find_optimal_k_loocv(X_features, y_np, k_range)

        self.perform_loocv_evaluation(X_features, y_np, self.optimal_k)


def main():
    # 假设文件在当前目录下
    id_file = '单链蛋白.txt'
    seq_file = '氨基酸序列（融合用）.txt'

    print("--- 开始数据加载与预处理 ---")
    dataset = PeptideDataset(id_file=id_file, seq_file=seq_file)
    print("\n原始数据预览:")
    print(dataset.get_dataframe().head())

    X = dataset.sequences
    y = dataset.labels

    print("\n--- 开始模型训练与评估 ---")
    classifier = ProteinClassifier()
    classifier.train_and_evaluate(X, y)
    print("\n--- 全流程执行完毕 ---")


if __name__ == '__main__':
    main()
