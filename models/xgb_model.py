import xgboost as xgb
import numpy as np
import os
import joblib # 使用joblib来保存和加载模型，比pickle更适合scikit-learn和xgboost模型

class XGBoostConsumePredictNet:
    def __init__(self, **xgb_params):
        """
        使用XGBoost参数来初始化模型。
        常见的参数有:
        - n_estimators: 树的数量，相当于DL中的epoch，但有early stopping后就不那么重要了
        - max_depth: 每棵树的最大深度，控制模型复杂度
        - learning_rate: 学习率，每次迭代的步长
        - subsample: 训练每棵树时，随机采样的训练数据比例
        - colsample_bytree: 训练每棵树时，随机采样的特征比例
        - objective: 目标函数, 'reg:squarederror' 用于回归
        """
        # 如果提供了GPU支持，则使用GPU
        try:
            # 检查是否有可用的GPU
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("XGBoost will use GPU.")
            else:
                self.device = 'cpu'
                print("XGBoost will use CPU.")
        except ImportError:
            self.device = 'cpu'
            print("PyTorch not found, XGBoost will use CPU.")

        # 设置默认参数，如果用户传入则覆盖
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000, # 先设置一个比较大的值，配合early stopping
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'device': self.device
        }
        # 用用户传入的参数更新默认参数
        default_params.update(xgb_params)

        self.model = xgb.XGBRegressor(**default_params)
        self.is_trained = False

    def fit(self, train_x, train_y, eval_x, eval_y, early_stopping_rounds=50):
        """
        训练模型。这个方法替换了你原来复杂的epoch循环。
        它接收全部的训练和验证数据。
        """
        print("Starting XGBoost model training...")
        # XGBoost的fit方法自带early stopping功能，非常方便
        # 它会在验证集上的性能不再提升时自动停止训练
        self.model.fit(
            train_x,
            train_y,
            eval_set=[(eval_x, eval_y)],
            # early_stopping_rounds=early_stopping_rounds,
            verbose=1000 # 每1000轮打印一次验证集损失
        )
        self.is_trained = True
        print("Training finished.")

    def predict(self, x):
        """
        进行预测。这个方法功能上等同于你原来的forward()方法。
        """
        if not self.is_trained:
            raise RuntimeError("You must train the model before making predictions!")
        # XGBoost的输入最好是numpy array
        if not isinstance(x, np.ndarray):
            # 假设输入是torch.Tensor，需要转换
             x = x.cpu().numpy()

        # 如果输入是一维的，需要reshape成(1, n_features)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return self.model.predict(x)

    def save_model(self, directory='pretrain', filename='xgb_model.json'):
        """
        保存模型。XGBoost有自己的模型保存格式。
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, filename)
        self.model.save_model(path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, directory='pretrain', filename='xgb_model.json'):
        """
        加载模型。这是一个类方法，可以直接通过类名调用。
        """
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        # 创建一个新的实例
        wrapper = cls()
        # 加载预训练的模型状态
        wrapper.model.load_model(path)
        wrapper.is_trained = True
        print(f"Model loaded from {path}")
        return wrapper