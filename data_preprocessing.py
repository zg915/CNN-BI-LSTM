import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def create_sequences(data, time_steps=10):
    """
    Create sequences of time_steps length and corresponding targets from standardized data.

    Parameters:
    data (array-like): Input two-dimensional standardized data (num_samples, num_features).
    time_steps (int): The size of the time step sequences.
    n_features (int): The number of features in each time step.

    Returns:
    X, y: Tuple of numpy arrays
          X is three-dimensional data of shape (None, time_steps, n_features) for the CNN-LSTM input.
          y is a one-dimensional array of targets, which are the next values following each sequence.
    """
    X = []
    for i in range(len(data) - time_steps + 1):
        X.append(data[i:i + time_steps, : ])
    
    return np.array(X)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def get_data(
            timestep = 10,
            batch_size = 64,
            y_name = 'Closing price',
            train_size = 6627,
            path = "Data/SSE000001.csv"):

    # Read data
    raw_df = pd.read_csv(path, index_col=0)

    # Split X into train and test
    X_train = raw_df.iloc[:train_size]
    X_test = raw_df.iloc[train_size:]

    # Standardize the data
    X_scaler = StandardScaler()
    X_scaler.fit(X_train)
    X_train_stand = X_scaler.transform(X_train)
    X_test_stand = X_scaler.transform(X_test)

    # Convert into time-sequence
    X_train_stand = create_sequences(X_train_stand)
    X_test_stand = create_sequences(X_test_stand)[: -1, : , : ]

    # Split y into train and test
    y_train = raw_df[y_name].iloc[timestep : train_size + 1]
    y_test = raw_df[y_name].iloc[train_size + timestep :]
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # Standardize the data
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    y_train_stand = y_scaler.transform(y_train)
    y_test_stand = y_scaler.transform(y_test)

    X_train = torch.tensor(X_train_stand).float()
    y_train = torch.tensor(y_train_stand).float()
    X_test = torch.tensor(X_test_stand).float()
    y_test = torch.tensor(y_test_stand).float()

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return X_train, y_train, X_test, y_test, y_scaler, train_loader, test_loader

if __name__ == "__main__":
    print(get_data())