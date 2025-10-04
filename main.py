import argparse
import pandas as pd
import numpy as np
import os
import wandb

# 假设 XGBoostConsumePredictNet 类定义在名为 xgb_model.py 的文件中
from models.xgb_model import XGBoostConsumePredictNet
from models.ridge_model import RidgeConsumePredictNet

from shap_analyzer import SHAPAnalyzer 

def main(args):

    print("Loading and preprocessing data...")
    df = pd.read_csv(args.raw_data_path)
    # 假设你的csv有header，如果没有，需要手动指定列名
    # df.columns = ['age', 'male', ..., 'citycode2']

    # 清洗数据 (用你自己的逻辑)
    # preprocess.clean_invalid_line(df) # 你需要让你的清洗函数支持DataFrame
    df = df.fillna(df.mean())  # 用均值填充

    # 使用One-Hot Encoding处理 citycode2
    df = pd.get_dummies(df, columns=['citycode2'], prefix='city')

    feature_names = df.drop('ln_pce', axis=1).columns.tolist()

    # 分离特征 (X) 和目标 (Y)
    Y = df['ln_pce'].values
    X = df.drop('ln_pce', axis=1).values

    # 定义数据集大小
    TRAIN_SET = int(len(X) * 0.7)
    VALID_SET = int(len(X) * 0.15)

    # 数据集划分 (切记，现在是Numpy arrays)
    train_x, train_y = X[:TRAIN_SET], Y[:TRAIN_SET]
    valid_x, valid_y = X[TRAIN_SET:TRAIN_SET + VALID_SET], Y[TRAIN_SET:TRAIN_SET + VALID_SET]
    test_x, test_y = X[TRAIN_SET + VALID_SET:], Y[TRAIN_SET + VALID_SET:]

    print(f"Data split: Train={len(train_x)}, Validation={len(valid_x)}, Test={len(test_x)}")
    print(f"Number of features after one-hot encoding: {train_x.shape[1]}")

    # 模型训练
    # 初始化模型，传入相关超参数
    if args.model_type == 'xgboost':
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
        }
        predict_net = XGBoostConsumePredictNet(**model_params)
    elif args.model_type == 'ridge':
        model_params = {
            'alpha': args.alpha
        }
        predict_net = RidgeConsumePredictNet(**model_params)

    # **核心改动：训练过程极其简单**
    # 整个训练循环被这一行替换了！
    predict_net.fit(train_x, train_y, valid_x, valid_y)
    predict_net.save_model()

    # 3. 在测试集上评估
    print('\n评估测试集:')
    total_bias = 0
    for i in range(len(test_x)):
        line_x = test_x[i]
        origin_y = test_y[i]

        # pred_y是一个只有一个元素np.array，用.item()获取数值
        pred_y = predict_net.predict(line_x).item()

        bias = pred_y - origin_y
        total_bias += abs(bias)
        # print(f'原始数据为{origin_y:.4f}, 模型输出为{pred_y:.4f}, 偏差为{bias:.4f}')
    test_MAE = total_bias / len(test_x)
    print(f"\n测试集平均绝对误差 (MAE): {test_MAE}")
    wandb.log({"test_MAE": test_MAE})

    # SHAP 分析
    if args.model_type == 'xgboost':
        print("\n" + "="*20)
        print("PERFORMING SHAP ANALYSIS")
        print("="*20)

        # 我们用训练集来分析模型学到的整体模式
        # 直接使用预处理后的DataFrame切片，避免numpy数组转换导致的类型问题
        train_x_df = df[:TRAIN_SET].drop('ln_pce', axis=1)

        # 假设你的 `predict_net` 对象中，训练好的模型实例存储在 `self.model`
        # 例如，在 XGBoostConsumePredictNet 的 fit 方法最后一行是 self.model = xgb.train(...)
        if hasattr(predict_net, 'model'):
            analyzer = SHAPAnalyzer(model=predict_net.model, data_df=train_x_df)

            # 生成并显示全局特征重要性图 (蜜蜂图)
            # 这是最有价值的图之一
            analyzer.plot_summary(plot_type='dot')

            # 你也可以看看传统的条形图
            analyzer.plot_summary(plot_type='bar')

            # --- 探索你最关心的特征 ---
            # 探索核心变量 'ifsocial' 的影响
            # 这会显示缴纳社保(ifsocial=1)与否，对模型预测值的SHAP贡献
            analyzer.plot_dependence('ifsocial')

            # 探索工资 'ln_wage' 的影响，并观察它是否与年龄 'age' 有交互作用
            analyzer.plot_dependence('ln_wage', interaction_feature='age')

            # 解释第一条训练数据的预测过程
            analyzer.explain_single_prediction(index=0)

            print("\n所有SHAP可视化已记录到wandb，请在wandb面板中查看。")

        else:
            print("错误: 'predict_net' 对象没有找到 'model' 属性。无法进行SHAP分析。")
            print("请确保在你的模型类中将训练好的模型保存在 self.model 中。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script with command line arguments.")
    parser.add_argument("--model_type", default="xgboost", choices=["xgboost", "ridge"], help="Type of model to use")
    parser.add_argument("--raw_data_path", "-p", help="Input file path", default="data/mydata.csv")
    parser.add_argument("--n_estimators", default=1000, type=int, help="Number of boosting rounds")
    parser.add_argument("--max_depth", default=4, type=int, help="Max tree depth")
    parser.add_argument("--learning_rate", "-lr", default=0.05, type=float, help="learning rate")

    # Ridge
    parser.add_argument("--alpha", default=1.0, type=float, help="Regularization strength for Ridge")

    args = parser.parse_args()
    run = wandb.init(
        entity="HCCS",
        # Set the wandb project where this run will be logged.
        project="SocialSecurity",
        # Track hyperparameters and run metadata.
        config=args,
    )
    main(args)