import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from copy import deepcopy as dc
from torch.utils.data import Dataset, DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Make a dataset for pytorch
class Stock_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# LSTM Class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(
            self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(
            self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Training_LSTM:
    def __init__(self,
                 df,
                 window_size,
                 split_size,
                 batch_size,
                 scaler,
                 dates_col,
                 vals_col,
                 input_size,
                 hidden_size,
                 num_stacked_layers,
                 device):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.split_size = split_size
        self.batch_size = batch_size
        self.scaler = scaler
        self.dates_col = dates_col
        self.vals_col = vals_col
        self.device = device
        self.df_LSTM = self.prepare_dataframe_for_lstm()
        self.last_series = self.df_LSTM.tail().to_numpy()[-1:][:, -1].item()
        self.df_np = self.df_LSTM.to_numpy()
        self.X, self.y = self.x_y_split()
        (self.X_train,
         self.X_test,
         self.y_train,
         self.y_test) = self.train_test_split()
        self.train_dataset = Stock_Dataset(self.X_train, self.y_train)
        self.test_dataset = Stock_Dataset(self.X_test, self.y_test)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False)
        self.model = LSTM(input_size, hidden_size, num_stacked_layers)
        self.model.to(device)

    def prepare_dataframe_for_lstm(self):
        df = dc(self.df)
        df.set_index(self.dates_col, inplace=True)
        for i in range(1, self.window_size+1):
            df[f'{self.vals_col}(t-{i})'] = df[self.vals_col].shift(i)
        df.dropna(inplace=True)
        return df[df.columns[::-1]]

    def x_y_split(self):
        df = self.scaler.fit_transform(self.df_np)
        X = df[:, :-1]
        y = df[:, -1]
        return X, y

    def train_test_split(self):
        train_size = int(len(self.X) * self.split_size)
        X_train = torch.tensor(
            self.X[:train_size].reshape((-1, self.window_size, 1))).float()
        X_test = torch.tensor(
            self.X[train_size:].reshape((-1, self.window_size, 1))).float()
        y_train = torch.tensor(
            self.y[:train_size].reshape((-1, 1))).float()
        y_test = torch.tensor(self.y[train_size:].reshape((-1, 1))).float()
        return X_train, X_test, y_train, y_test

    def training(self, lr=0.01, epochs=5, verbose=False):
        loss_fun = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_train_hist = list()
        loss_val_hist = list()
        for epoch in range(epochs):
            self.model.train(True)
            if verbose:
                print(f'Epoch: {epoch + 1}')
            running_loss = 0.0

            for batch_index, batch in enumerate(self.train_loader):
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)

                output = self.model(x_batch)
                loss = loss_fun(output, y_batch)
                running_loss += loss.item()
                loss_train_hist.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_index % 100 == 99:  # print every 100 batches
                    avg_loss_across_batches = running_loss / 100
                    if verbose:
                        print('Batch {0}, Loss: {1:.3f}'.format(
                            batch_index+1, avg_loss_across_batches))
                    running_loss = 0.0

            if verbose:
                print()
            # Validating
            self.model.train(False)
            running_loss = 0.0

            for batch_index, batch in enumerate(self.test_loader):
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)

                with torch.no_grad():
                    output = self.model(x_batch)
                    loss = loss_fun(output, y_batch)
                    running_loss += loss.item()
                    loss_val_hist.append(loss.item())
            avg_loss_across_batches = running_loss / len(self.test_loader)

            if verbose:
                print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
                print('***************************************************')
                print()

        if verbose:
            _, axes = plt.subplots(ncols=2, figsize=(10, 5))
            axes[0].plot(loss_train_hist)
            axes[0].set_title('Train Loss')
            axes[1].plot(loss_val_hist)
            axes[1].set_title('Validation Loss')
            for ax in axes:
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE')
            plt.tight_layout()
            plt.show()
        # return loss_train_hist, loss_val_hist

    def convert_vals(self, X_vals, y_vals, pred=False):
        trans_vals = np.zeros((X_vals.shape[0], self.window_size+1))
        if pred:
            with torch.no_grad():
                predicted = self.model(
                    X_vals.to(self.device)).to('cpu').numpy()
            trans_vals[:, 0] = predicted.flatten()
        else:
            trans_vals[:, 0] = y_vals.flatten()
        trans_vals = self.scaler.inverse_transform(trans_vals)
        return dc(trans_vals[:, 0])

    def plot_pred(self):
        _, axes = plt.subplots(ncols=2, figsize=(10, 5))
        axes[0].plot(self.convert_vals(self.X_train, self.y_train),
                     label=f'Real {self.vals_col}')
        axes[0].plot(self.convert_vals(self.X_train, self.y_train, True),
                     label=f'Predicted {self.vals_col}')
        axes[0].set_title("Training prediction vs real values")
        axes[1].plot(self.convert_vals(self.X_test, self.y_test),
                     label=f'Real {self.vals_col}')
        axes[1].plot(self.convert_vals(self.X_test, self.y_test, True),
                     label=f'Predicted {self.vals_col}')
        axes[1].set_title("Validation prediction vs real values")
        for ax in axes:
            ax.set_xlabel('Day')
            ax.set_ylabel(f'{self.vals_col}')
        plt.legend()
        plt.show()

    def pred_vals(self):
        new_pred = torch.cat(
            (self.X_test[-1].flatten(), self.y_test[-1]))[1:].reshape_as(
                self.X_test[-1])
        val_to_pred = torch.cat(
            (self.X_test[-1], new_pred)).reshape(2, self.window_size, 1)
        with torch.no_grad():
            val_test = self.model(val_to_pred)[-1].numpy().reshape(-1, 1)
        trans_vals = np.zeros((val_test.shape[0], self.window_size+1))
        trans_vals[:, 0] = val_test.flatten()
        pred_val = self.scaler.inverse_transform(trans_vals)[:, 0].item()
        return pred_val
