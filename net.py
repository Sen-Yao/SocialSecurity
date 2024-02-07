import time

import torch
import torch.nn as nn


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class ConsumePredictNet(nn.Module):
    def __init__(self, lr, gamma, num_city_codes, embedding_dim=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(num_city_codes, embedding_dim)
        self.layer1 = nn.Sequential(
            nn.Linear(16 + embedding_dim - 1, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 2048), nn.ReLU(),
            nn.Linear(2048, 8192)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(8192, 8192), nn.ReLU(),
            nn.Linear(8192, 8192), nn.ReLU(),
            nn.Linear(8192, 8192)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(8192, 2048), nn.ReLU(),
            nn.Linear(2048, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.to(try_gpu())
        # 使用Xavier初始化
        self.apply(self.xavier_init)
        self.learning_rate = float(lr)
        self.gamma = gamma
        self.trainer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.trainer, gamma=self.gamma)
        self.loss = 0
        self.epoch = 0
        self.avg_loss = 0
        self.update_in_epoch = 0
        self.last_output = time.time()
        self.lambda_reg = 0.01

    def forward(self, x):
        # 假设城市代码是最后一个特征
        city_code = x[:, -1].long()
        x = x[:, :-1]  # 删除原始城市代码
        city_code_embed = self.embedding(city_code)
        x = torch.cat([x, city_code_embed], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def InitNet(self):
        # Initial net
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.1)
        self.apply(init_weights)

    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def Update(self, predict_y, batch_y):
        loss = self.loss_function(predict_y, batch_y)
        self.loss = loss
        self.avg_loss += loss
        self.update_in_epoch += 1
        self.trainer.zero_grad()
        loss.backward()
        self.trainer.step()

    def loss_function(self, y_true, y_pred):
        mse_loss = nn.HuberLoss()
        mse = mse_loss(y_true, y_pred)
        l2_regularization = torch.tensor(0.0).to(try_gpu())
        for param in self.parameters():
            l2_regularization += torch.norm(param, p=2)
        return mse
        # return mse + self.lambda_reg * l2_regularization

    def output(self, pred_valid_y, true_valid_y):
        # 计算预测偏差的绝对值
        absolute_differences = torch.abs(pred_valid_y - true_valid_y)

        # 计算正确的预测数（差的绝对值小于等于0.3）
        correct_predictions = (absolute_differences <= 0.3).float().sum()

        # 计算正确率
        accuracy = 100 * correct_predictions / pred_valid_y.numel()
        print(f'epoch={self.epoch}, speed={1 / (time.time() - self.last_output):.2f} epoch/s, '
              f'lr={self.trainer.state_dict()["param_groups"][0]["lr"]:.2e}, '
              f'loss={(self.avg_loss / self.update_in_epoch).item():.4f}, '
              f'valid bias={abs(true_valid_y - pred_valid_y).mean().item():.4f}, '
              f''f'accuracy={accuracy.item():.2f}%'
              )
        self.last_output = time.time()
