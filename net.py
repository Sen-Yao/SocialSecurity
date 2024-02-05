import torch
import torch.nn as nn


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class ConsumePredictNet:
    def __init__(self, lr):
        self.net = nn.Sequential(
            nn.Linear(15, 32), nn.ReLU(),
            nn.Linear(32, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 2), nn.ReLU(),
            nn.Linear(2, 1),
        )
        self.net.to(try_gpu())
        self.InitNet()
        self.learning_rate = float(lr)
        self.trainer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss = None
        self.lambda_reg = 0.01

    def InitNet(self):
        # Initial net
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        self.net.apply(init_weights)

    def Update(self, predict_y, batch_y):
        loss = self.loss_function(predict_y, batch_y)
        self.loss = loss
        self.trainer.zero_grad()
        loss.backward()
        self.trainer.step()

    def loss_function(self, y_true, y_pred):
        mse_loss = nn.MSELoss()
        mse = mse_loss(y_true, y_pred)
        l2_regularization = torch.tensor(0.0).to(try_gpu())
        for param in self.net.parameters():
            l2_regularization += torch.norm(param, p=2)
        return mse
        # return mse + self.lambda_reg * l2_regularization
