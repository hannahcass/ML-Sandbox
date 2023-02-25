import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data_processing import DataPreProcessing, DataSplit
from models import rf, lr, knn, dt, xg, LSTMRegression

from training import train_vgg16, train_lstm


data = pd.read_csv(
    "D:\projects_2023\main\datasets\mobile_price_prediction_kaggle\Cellphone.csv")

data_preprocessed = DataPreProcessing(data).main(threshold=0.4)
x_train, x_test, y_train, y_test = DataSplit(
    data_preprocessed, "Price", 42, 0.2).main()


# results = []
# for model in [rf, lr, knn, dt, xg]:
#     print(f"Running model: {model.__class__.__name__}")
#     model.fit(x_train, y_train)
#     train_acc = model.score(x_train, y_train)
#     test_acc = model.score(x_test, y_test)
#     results.append({'Model': model.__class__.__name__,
#                    'Training Accuracy': train_acc, 'Testing Accuracy': test_acc})

# results_df = pd.DataFrame(results)


# LSTM
epochs = 10
batch_size = 32
lr = 0.001

input_size = 13
hidden_size = 64
output_size = 1
num_layers = 1


model = LSTMRegression(input_size, hidden_size,
                       num_layers, output_size)

loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

train_data = []
for i in range(len(x_train)):
    train_data.append((torch.tensor(x_train[i]).float(), torch.tensor(
        [y_train[i]]).float()))

train_loader = torch.utils.data.DataLoader(
    train_data, shuffle=True, batch_size=batch_size)


train_loss, test_loss = train_lstm(
    model, train_loader, loss, optimizer, epochs)


print("Training loss:", train_loss)
print("Test loss:", test_loss)


# for param in vgg16.features.parameters():
#     param.requires_grad = False

# num_features = vgg16.classifier[6].in_features
# vgg16.classifier[6] = nn.Linear(25088, 1)

# train_acc, test_acc = train_vgg16(
#     vgg16, train_loader, loss, optimizer, epochs, batch_size)
# results.append({'Model': 'VGG-16', 'Training Accuracy': train_acc,
#                 'Testing Accuracy': test_acc})


# print(results_df)
