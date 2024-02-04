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
            nn.Linear(16, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, 4096), nn.ReLU(),
            nn.Linear(4096, 8192), nn.ReLU(),
            nn.Linear(8192, 8192), nn.ReLU(),
            nn.Linear(8192, 8192), nn.ReLU(),
            nn.Linear(8192, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.net.to(try_gpu())
        self.InitNet()
        self.loss_function = nn.MSELoss()
        self.learning_rate = float(lr)
        self.trainer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss = None

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
