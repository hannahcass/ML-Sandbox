from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import torch
import torchvision.models as models
import torch.nn as nn


from params import params

rf = RandomForestRegressor()
lr = LinearRegression()
knn = KNeighborsRegressor()
dt = DecisionTreeRegressor()
xg = XGBRegressor(**params)


#vgg16 = models.vgg16(pretrained=True)


class LSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size, device=x.device)
        out, _ = self.lstm(x.unsqueeze(0), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(1)


 