import re
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import itertools
import warnings

warnings.filterwarnings('ignore')

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
    def __init__(self, ids_file='单链蛋白.txt', seq_file='氨基酸序列（融合用）.txt'):
        try:
            # 兼容旧的文件名
            if ids_file == '单链蛋白.txt':
                ids_file = '单链蛋白.txt'
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


class PCPseAACFeatureExtractor:
    def __init__(self, w=0.1, lambda_param=5):
        self.w = w
        self.lambda_param = lambda_param
        self.amino_acid_properties = amino_acid_properties
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    def extract_features(self, sequences):
        features = []
        for sequence in sequences:
            if not sequence: continue

            # 氨基酸组成频率
            aa_freq = np.array([sequence.count(aa) / len(sequence) for aa in self.amino_acids])

            # 物理化学特征
            properties = list(next(iter(self.amino_acid_properties.values())).keys())
            phys_features_list = []
            for prop in properties:
                prop_values = [self.amino_acid_properties.get(aa, {}).get(prop, 0) for aa in sequence]
                phys_features_list.extend(
                    [np.mean(prop_values), np.std(prop_values), np.max(prop_values), np.min(prop_values)])
            phys_features = np.array(phys_features_list)

            # 序列相关性
            correlations = []
            L = len(sequence)
            selected_props = ['polarity', 'hydrophobicity', 'charge', 'secondary_structure']
            for k in range(1, self.lambda_param + 1):
                correlation = 0
                if L > k:
                    for i in range(L - k):
                        aa1_props = self.amino_acid_properties.get(sequence[i], {})
                        aa2_props = self.amino_acid_properties.get(sequence[i + k], {})
                        prop_corr = np.sqrt(
                            np.sum([(aa1_props.get(prop, 0) - aa2_props.get(prop, 0)) ** 2 for prop in selected_props]))
                        correlation += prop_corr
                    correlations.append(correlation / (L - k))
                else:
                    correlations.append(0)
            seq_corr = np.array(correlations)

            w_float = float(self.w)
            balanced_feature = np.concatenate([(1 - w_float) * aa_freq, phys_features, w_float * seq_corr])
            features.append(balanced_feature)

        return np.array(features)


class ModelEvaluator:
    @staticmethod
    def perform_loocv_evaluation(classifier, X, y):
        print("\n===== 留一交叉验证（LOOCV）性能评估 =====")
        y_pred_all = []
        y_true_all = []
        loo = LeaveOneOut()

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

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
        print(classification_report(y_true_all, y_pred_all))

        return {
            'accuracy': accuracy_score(y_true_all, y_pred_all),
            'precision': precision_score(y_true_all, y_pred_all),
            'recall': recall_score(y_true_all, y_pred_all),
            'f1_score': f1_score(y_true_all, y_pred_all)
        }


class ProteinClassifier:
    def __init__(self, random_state=42):
        self.feature_extractor = PCPseAACFeatureExtractor()
        self.scaler = StandardScaler()
        self.evaluator = ModelEvaluator()
        self.optimal_params = None
        self.random_state = random_state

    def find_optimal_lr_params_loocv(self, X_scaled, y, param_grid):
        """
        使用LOOCV和网格搜索寻找逻辑回归的最佳超参数
        """
        print(f"\n===== 开始通过LOOCV寻找逻辑回归最佳超参数 =====")
        best_score = -1
        best_params = {}
        y = np.array(y)

        # *** 修改部分开始 ***
        # 直接从传入的字典中获取参数列表
        solvers = param_grid['solver']
        penalties = param_grid['penalty']
        Cs = param_grid['C']

        # 生成所有参数组合
        all_param_combinations = list(itertools.product(Cs, penalties, solvers))

        for C, penalty, solver in all_param_combinations:
            # 跳过无效的求解器和正则化组合
            if solver == 'liblinear' and penalty not in ['l1', 'l2']:
                continue
            if solver == 'saga' and penalty not in ['l1', 'l2']:
                continue

            current_params = {'C': C, 'penalty': penalty, 'solver': solver}
            lr = LogisticRegression(**current_params, random_state=self.random_state, max_iter=2000)

            try:
                scores = cross_val_score(lr, X_scaled, y, cv=LeaveOneOut(), scoring='accuracy', error_score='raise')
                mean_score = scores.mean()
                print(f"测试参数: {current_params}, LOOCV 平均准确率: {mean_score:.4f}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = current_params
            except Exception as e:
                print(f"参数组合 {current_params} 遇到错误: {e}")
                continue
        # *** 修改部分结束 ***

        print(f"\n--- 搜索完成 ---")
        print(f"找到的最佳参数组合: {best_params} (准确率: {best_score:.4f})")
        return best_params

    def train_and_evaluate(self, X, y):
        # 1. 特征提取和标准化
        print("\n===== 步骤1：特征提取与数据标准化 =====")
        X_features = self.feature_extractor.extract_features(X)
        X_scaled = self.scaler.fit_transform(X_features)
        y = np.array(y)

        # 2. 定义参数网格并通过LOOCV寻找最佳参数
        param_grid_for_search = {
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100]
        }

        self.optimal_params = self.find_optimal_lr_params_loocv(X_scaled, y, param_grid_for_search)

        if not self.optimal_params:
            print("错误：未能找到最佳逻辑回归参数，终止评估。")
            return

        # 3. 使用最佳参数创建最终模型，并用LOOCV进行评估
        print(f"\n===== 步骤2：使用最佳参数 {self.optimal_params} 进行最终评估 =====")
        final_classifier = LogisticRegression(
            **self.optimal_params,
            random_state=self.random_state,
            max_iter=2000
        )

        cv_results = self.evaluator.perform_loocv_evaluation(
            final_classifier, X_scaled, y
        )

        return final_classifier, cv_results


def main():
    # 兼容旧的文件名
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

    classifier = ProteinClassifier(random_state=42)
    classifier.train_and_evaluate(X, y)


if __name__ == '__main__':
    main()