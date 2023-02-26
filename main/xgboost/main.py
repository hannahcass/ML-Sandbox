import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import optuna

from data_processing import DataPreProcessing, DataSplit, DataManipulation
from models import rf, lr, knn, dt, xg, LSTMRegression

from training import train_vgg16, train_lstm


# data = pd.read_csv(
#    "D:\projects_2023\main\datasets\mobile_price_prediction_kaggle\Cellphone.csv")

data = pd.read_csv(
    "D:\projects_2023\main\datasets\stock_exchange_data\indexProcessed.csv")

data_preprocessed = DataPreProcessing(data).main(threshold=0.4)

data_GDAXI = DataManipulation(data_preprocessed).filter_rows_by_index("GDAXI")

data_GDAXI = data_GDAXI.drop(["Index", "Date"], axis=1)
x_train, x_test, y_train, y_test = DataSplit(
    data_GDAXI, "CloseUSD", 42, 0.2).main()


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
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
batch_size = 64


train_data = []
for i in range(len(x_train)):
    train_data.append((torch.tensor(x_train[i]).float(), torch.tensor(
        [y_train[i]]).float()))

train_loader = torch.utils.data.DataLoader(
    train_data, shuffle=True, batch_size=batch_size)


test_data = []
for i in range(len(x_test)):
    test_data.append((torch.tensor(x_test[i]).float(), torch.tensor(
        [y_test[i]]).float()))

test_loader = torch.utils.data.DataLoader(
    test_data, shuffle=True, batch_size=batch_size)



def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 118, 128, log=True)
    num_layers = trial.suggest_int('num_layers', 4, 6)
    lr = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    epochs = trial.suggest_int('epochs', 10, 100)

    input_size = 6
    #hidden_size = 64
    output_size = 1
    #num_layers = 2






    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = LSTMRegression(input_size, hidden_size,
                       num_layers, output_size).to(device)

    loss = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=float(lr))


    train_loss, test_loss = train_lstm(
        model, train_loader, test_loader, loss, optimizer, epochs)
    
    return train_loss, test_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


# for param in vgg16.features.parameters():
#     param.requires_grad = False

# num_features = vgg16.classifier[6].in_features
# vgg16.classifier[6] = nn.Linear(25088, 1)

# train_acc, test_acc = train_vgg16(
#     vgg16, train_loader, loss, optimizer, epochs, batch_size)
# results.append({'Model': 'VGG-16', 'Training Accuracy': train_acc,
#                 'Testing Accuracy': test_acc})


# print(results_df)
