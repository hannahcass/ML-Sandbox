import torch
import numpy as np

##from models import vgg16
from models import LSTMRegression










def train_lstm(model, train_loader, test_loader, loss, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            l = loss(output, batch_y)
            l.backward()
            optimizer.step()
            train_loss += l.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        print("Epoch: {} Training Loss: {:.6f}".format(epoch+1, train_loss))

        model.eval()
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with torch.no_grad():
                output = model(batch_x)
                l = loss(output, batch_y)
                test_loss += l.item() * batch_x.size(0)
        test_loss /= len(test_loader.dataset)
        print("Test loss: {:.6f}".format(test_loss))

    return train_loss, test_loss







def train_vgg16(model, train_loader, loss, optimizer, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            output = model(batch_x)
            loss = loss(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        model.eval()
        train_preds = []
        train_targets = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).float()
            output = model(batch_x)
            train_preds.append(output.squeeze().cpu().numpy())
            train_targets.append(batch_y.cpu().numpy())
        train_loss = loss(
            np.concatenate(train_preds), np.concatenate(train_targets))

        test_preds = []
        test_targets = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device).float()
            output = model(batch_x)
            test_preds.append(output.squeeze().cpu().numpy())
            test_targets.append(batch_y.cpu().numpy())
        test_loss = loss(
            np.concatenate(test_preds), np.concatenate(test_targets))

    return train_loss, test_loss
