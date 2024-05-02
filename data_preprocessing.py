import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def create_sequences(data, timestep=10):
    """
    Create sequences of timestep length and corresponding targets from standardized data.

    Parameters:
    data (array-like): Input two-dimensional standardized data (num_samples, num_features).
    timestep (int): The size of the time step sequences.
    n_features (int): The number of features in each time step.

    Returns:
    X, y: Tuple of numpy arrays
          X is three-dimensional data of shape (None, timestep, n_features) for the CNN-LSTM input.
          y is a one-dimensional array of targets, which are the next values following each sequence.
    """
    X = []
    for i in range(len(data) - timestep + 1):
        X.append(data[i:i + timestep, : ])
    
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
            train_size = 0.9,
            path = "Data/SSE000001.csv",
            split_method = "time",
            split_number = 10):

    # Read data
    raw_df = pd.read_csv(path, index_col=0)

    if train_size <= 1:
        train_size = int(train_size * len(raw_df))

    # Split X, y into train and test  
    if split_method == "time":
        X_train = raw_df.iloc[:train_size]
        X_test = raw_df.iloc[train_size:]
        y_train = raw_df[y_name].iloc[timestep : train_size + 1]
        y_test = raw_df[y_name].iloc[train_size + timestep :]

        # Standardize the data
        X_scaler = StandardScaler()
        X_scaler.fit(X_train)
        X_train_stand = X_scaler.transform(X_train)
        X_test_stand = X_scaler.transform(X_test)

        # Convert into time-sequence
        X_train_stand = create_sequences(X_train_stand, timestep = timestep)
        X_test_stand = create_sequences(X_test_stand, timestep = timestep)[: -1, : , : ]

        # Split y into train and test
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
    
    elif split_method == "uniform":
        part_size = len(raw_df) // split_number
        part_train_size = train_size // split_number
        # List to store each part
        X_trains = []
        X_tests = []
        y_trains = []
        y_tests = []
        # Splitting the DataFrame into parts
        for i in range(split_number):
            start_index = i * part_size
            if i == split_number - 1:  # Handle the last part to include the remainder
                end_index = len(raw_df)
            else:
                end_index = start_index + part_size
            part_df = raw_df.iloc[start_index:end_index]
            
            X_trains.append(part_df.iloc[:part_train_size])
            X_tests.append(part_df.iloc[part_train_size:])
            y_trains.append(part_df[y_name].iloc[timestep : part_train_size + 1])
            y_tests.append(part_df[y_name].iloc[part_train_size + timestep :])


    # Standardize the data
    X_scaler = StandardScaler()
    X_scaler.fit(pd.concat(X_trains))
    X_trains_stand = []
    X_tests_stand = []
    for i in range(split_number):
        X_trains_stand.append(X_scaler.transform(X_trains[i]))
        X_tests_stand.append(X_scaler.transform(X_tests[i]))

        X_trains_stand[-1] = create_sequences(X_trains_stand[i], timestep = timestep)
        X_tests_stand[-1] = create_sequences(X_tests_stand[i], timestep = timestep)[: -1, : , : ]

    X_train_stand = np.concatenate(X_trains_stand, axis = 0)
    X_test_stand = np.concatenate(X_tests_stand, axis = 0)

    # Split y into train and test
    y_train = pd.concat(y_trains)
    y_test = pd.concat(y_tests)
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
    X_train, y_train, X_test, y_test, y_scaler, train_loader, test_loader = get_data(
            timestep = 10,
            batch_size = 64,
            y_name = 'Closing price',
            train_size = 0.9,
            path = "Data/SSE000001.csv",
            split_method = "uniform",
            split_number = 10)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)