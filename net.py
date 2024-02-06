import torch
import torch.nn as nn


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class ConsumePredictNet(nn.Module):
    def __init__(self, lr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Sequential(
            nn.Linear(15, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 2048), nn.ReLU(),
            nn.Linear(2048, 8192), nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(8192, 8192), nn.ReLU(),
            nn.Linear(8192, 8192), nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(8192, 2048), nn.ReLU(),
            nn.Linear(2048, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 8), nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.to(try_gpu())
        self.InitNet()
        self.learning_rate = float(lr)
        self.trainer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.trainer, gamma=0.999)
        self.loss = None
        self.lambda_reg = 0.01

    def forward(self, x):
        x = self.layer1(x)
        # print('layer1 = ', x)
        x = self.layer2(x)
        # print('layer2 = ', x)
        x = self.layer3(x)
        # print('layer3 = ', x)
        return x

    def InitNet(self):
        # Initial net
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.06)
        self.apply(init_weights)

    def Update(self, predict_y, batch_y):
        loss = self.loss_function(predict_y, batch_y)
        self.loss = loss
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
