# file: models/ridge_model.py

import os
import joblib  # 使用joblib来保存scikit-learn模型
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class RidgeConsumePredictNet:
    """
    一个封装了StandardScaler和Ridge回归的类，接口与XGBoostConsumePredictNet保持一致。
    """
    def __init__(self, alpha=1.0, **kwargs):
        """
        初始化模型。
        :param alpha: Ridge回归的正则化强度。alpha越大，正则化越强，系数越趋向于0。
        """
        # 1. 初始化一个StandardScaler，用于特征标准化
        self.scaler = StandardScaler()
        
        # 2. 初始化Ridge回归模型
        #    - alpha是需要调整的关键超参数
        #    - random_state保证结果可复现
        self.model = Ridge(alpha=alpha, random_state=42)
        
        self.is_trained = False
        print(f"RidgeConsumePredictNet initialized with alpha={alpha}")

    def fit(self, train_x, train_y, eval_x=None, eval_y=None):
        """
        训练模型。这个方法会先拟合标准化器，然后用标准化后的数据训练岭回归模型。
        为了保持接口一致性，eval_x和eval_y参数会被接收但不会被使用，因为scikit-learn的Ridge模型没有内置early stopping。
        """
        print("Starting Ridge model training...")
        
        # 关键第一步：在训练数据上拟合StandardScaler
        self.scaler.fit(train_x)
        
        # 关键第二步：使用拟合好的scaler来转换训练数据
        train_x_scaled = self.scaler.transform(train_x)
        
        # 第三步：在标准化后的数据上训练Ridge模型
        self.model.fit(train_x_scaled, train_y)
        
        self.is_trained = True
        print("Training finished.")

    def predict(self, x):
        """
        进行预测。
        预测时，必须使用在训练集上拟合的同一个StandardScaler来转换新数据。
        """
        if not self.is_trained:
            raise RuntimeError("You must train the model before making predictions!")
        
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        # 如果输入是一维的，需要reshape成(1, n_features)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # 关键步骤：使用训练时生成的scaler来转换预测数据
        x_scaled = self.scaler.transform(x)
        
        return self.model.predict(x_scaled)

    def save_model(self, directory='pretrain', filename='ridge_model.joblib'):
        """
        保存模型和标准化器。
        使用joblib来打包保存多个scikit-learn对象。
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        path = os.path.join(directory, filename)
        
        # 将模型和scaler打包到一个字典中进行保存
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"Model and scaler saved to {path}")
        
    @classmethod
    def load_model(cls, directory='pretrain', filename='ridge_model.joblib'):
        """
        加载模型和标准化器。
        """
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
            
        # 加载包含模型和scaler的字典
        saved_objects = joblib.load(path)
        
        # 创建一个新的类实例
        wrapper = cls(alpha=saved_objects['model'].alpha)
        
        # 将加载的对象赋值给实例
        wrapper.model = saved_objects['model']
        wrapper.scaler = saved_objects['scaler']
        wrapper.is_trained = True
        
        print(f"Model and scaler loaded from {path}")
        return wrapper