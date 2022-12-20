from keras.layers import LSTM, Dense, Dropout, CuDNNLSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

class PricePrediction:
    """A Class utility to train and predict price of stocks/cryptocurrencies/trades
        using keras model"""
    def __init__(self, ticker_name, **kwargs):
        """
        :param ticker_name (str): ticker name, e.g. usd, btc, aapl, nflx, etc.
        :param n_steps (int): sequence length used to predict, default is 60
        :param verbose (int): verbosity level, default is 1
        ==========================================
        Model parameters
        :param n_layers (int): number of recurrent neural network layers, default is 3
        :param cell (keras.layers.RNN): RNN cell used to train keras model, default is LSTM
        :param units (int): number of units of `cell`, default is 256
        :param dropout (float): dropout rate ( from 0 to 1 ), default is 0.3
        ==========================================
        Training parameters
        :param batch_size (int): number of samples per gradient update, default is 64
        :param epochs (int): number of epochs, default is 100
        :param optimizer (str, keras.optimizers.Optimizer): optimizer used to train, default is 'adam'
        :param loss (str, function): loss function used to minimize during training,
            default is 'mae'
        :param test_size (float): test size ratio from 0 to 1, default is 0.15
        """
        self.ticker_name = ticker_name
        self.n_steps = kwargs.get("n_steps", 60)
        self.verbose = kwargs.get("verbose", 1)

        self.n_layers = kwargs.get("n_layers", 3)
        self.cell = kwargs.get("cell", LSTM)
        self.units = kwargs.get("units", 256)
        self.dropout = kwargs.get("dropout", 0.3)

        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 100)
        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "mae")
        self.test_size = kwargs.get("test_size", 0.15)
       
        # runtime attributes
        self.model_trained = False
      
    def create_model(self):
        """Construct and compile the keras model"""
        model = create_model(input_length=self.n_steps,
                                    units=self.units,
                                    cell=self.cell,
                                    dropout=self.dropout,
                                    n_layers=self.n_layers,
                                    loss=self.loss,
                                    optimizer=self.optimizer)


    def train(self, path):

        self.df = pd.read_csv(path)
        self.df['output_date'] = pd.to_datetime(self.df['output_date'])
        # set the index
        self.df.set_index('output_date', inplace=True)
        self.df.drop(["mkt_id"], axis=1, inplace=True)

        plt.figure(figsize=(18, 9))
        plt.title('Profit History')
        plt.plot(self.df['output_own_profits'])
        plt.xlabel('Date')
        plt.ylabel('Profits ($)')
        # plt.show()

        close_prices = self.df['output_own_profits']
        values = close_prices.values
        self.training_data_len = math.ceil(len(values)* 0.8)

        self.scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = self.scaler.fit_transform(values.reshape(-1,1))

        train_data = scaled_data[0: self.training_data_len, :]

        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        test_data = scaled_data[self.training_data_len-60: , : ]
        self.x_test = []
        self.y_test = values[self.training_data_len:]

        for i in range(60, len(test_data)):
            self.x_test.append(test_data[i-60:i, 0])

        self.x_test = np.array(self.x_test)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

        # ltsm model 
        self.model = keras.Sequential()
        self.model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(layers.LSTM(100, return_sequences=False))
        self.model.add(layers.Dense(25))
        self.model.add(layers.Dense(1))
        self.model.summary()

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(x_train, y_train, batch_size= 1, epochs=3)

        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")


    def predict(self):
        predictions = self.model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - self.y_test)**2)
        rmse
        data = self.df.filter(['output_own_profits'])
        train = data[:self.training_data_len]
        validation = data[self.training_data_len:]
        validation['Predictions'] = predictions
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Profit USD ($)')
        plt.plot(train)
        plt.plot(validation[['output_own_profits', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()

    # some metrics

    def get_MAE(self):
        """Calculates the Mean-Absolute-Error metric of the test set"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call `model.train()` first.")
        y_pred, y_true = self.predict()

        return mean_absolute_error(y_true, y_pred)

    def get_MSE(self):
        """Calculates the Mean-Squared-Error metric of the test set"""
        if not self.model_trained:
            raise RuntimeError("Model is not trained yet, call `model.train()` first.")
        y_pred, y_true = self.predict()
        return mean_squared_error(y_true, y_pred)

    def plot_test_set(self):
        """Plots test data"""
        y_pred, y_true = self.predict()
        plt.plot(y_true, c='b')
        plt.plot(y_pred, c='r')
        plt.xlabel("Days")
        plt.ylabel("Profit USD ($)")
        plt.legend(["Actual Profit", "Predicted Profit"])
        plt.show()

