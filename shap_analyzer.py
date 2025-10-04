# shap_analyzer.py

import shap
import pandas as pd
import matplotlib.pyplot as plt

class SHAPAnalyzer:
    """
    一个用于解释和可视化机器学习模型的SHAP分析器。
    特别为像XGBoost这样的树模型优化。
    """
    def __init__(self, model, data_df: pd.DataFrame):
        """
        初始化分析器。

        参数:
            model: 已经训练好的模型对象 (例如, a raw xgboost.XGBRegressor instance)。
            data_df (pd.DataFrame): 用于进行SHAP分析的数据，特征名应为列名。
                                   通常使用训练集来了解模型的整体行为。
        """
        if not isinstance(data_df, pd.DataFrame):
            raise TypeError("输入数据 'data_df' 必须是 Pandas DataFrame 类型。")

        self.model = model
        self.data_df = data_df
        self.feature_names = data_df.columns.tolist()

        print("Initializing SHAP explainer...")
        # 对于XGBoost等树模型，TreeExplainer是最高效的
        self.explainer = shap.TreeExplainer(self.model)
        
        print("Calculating SHAP values... (This may take a moment)")
        # 计算SHAP值。返回的是一个Explanation对象，比原始numpy数组更易用
        self.shap_values = self.explainer(self.data_df)
        print("SHAP values calculated successfully.")

    def plot_summary(self, plot_type='dot'):
        """
        生成一个全局特征重要性图。

        参数:
            plot_type (str): 'dot' (默认) 生成散点图（又称蜜蜂图），展示特征影响的方向和分布。
                             'bar' 生成传统的条形图，只显示平均重要性。
        """
        print(f"\nGenerating summary plot (type: {plot_type})...")
        plt.title(f'SHAP Feature Importance ({plot_type} plot)')
        shap.summary_plot(self.shap_values, self.data_df, plot_type=plot_type, show=False)
        plt.tight_layout()
        plt.show()

    def plot_dependence(self, feature: str, interaction_feature: str = None):
        """
        生成一个依赖图，展示单个特征如何影响模型的预测。

        参数:
            feature (str): 要分析的主要特征的名称 (必须是data_df中的一列)。
            interaction_feature (str, optional): 用于给散点图着色的交互特征。
                                                 如果为None，SHAP会自动选择一个交互最强的特征。
        """
        if feature not in self.feature_names:
            raise ValueError(f"特征 '{feature}' 不在数据列中。")

        print(f"\nGenerating dependence plot for feature: '{feature}'...")
        
        # 为了更好的可视化，如果交互特征未指定，让SHAP自动选择
        interaction_idx = interaction_feature
        if interaction_feature:
            print(f"Coloring by interaction feature: '{interaction_feature}'")
        else:
            print("Interaction feature not specified, SHAP will choose automatically.")
            interaction_idx = "auto"

        plt.title(f'SHAP Dependence Plot for {feature}')
        shap.dependence_plot(
            feature,
            self.shap_values.values,
            self.data_df,
            interaction_index=interaction_idx,
            show=False
        )
        plt.tight_layout()
        plt.show()

    def explain_single_prediction(self, index: int):
        """
        为数据集中的单个样本生成一个力图 (force plot)，解释该样本的预测过程。
        注意：此图在Jupyter Notebook中为交互式。

        参数:
            index (int): 要解释的样本在data_df中的行索引。
        """
        if not (0 <= index < len(self.data_df)):
            raise IndexError(f"索引 {index} 超出范围。")
            
        print(f"\nGenerating force plot for prediction at index {index}...")
        
        # shap.initjs() # 如果在jupyter notebook中需要取消注释
        
        force_plot = shap.force_plot(
            self.explainer.expected_value,
            self.shap_values.values[index, :],
            self.data_df.iloc[index, :]
        )
        
        # 在脚本中运行时，需要特殊方式保存为html文件来查看
        try:
            shap.save_html(f"force_plot_index_{index}.html", force_plot)
            print(f"Force plot saved to 'force_plot_index_{index}.html'. Please open this file in a browser.")
        except Exception as e:
            print(f"Could not save force plot as HTML. Error: {e}")
            print("Displaying force plot requires a compatible environment like Jupyter notebooks.")