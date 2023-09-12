import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.callbacks import EarlyStopping


class RootCause:
    def __init__(self):
        self.MinMaxScaler = MinMaxScaler(feature_range=(-1, 1))


    def create_dataset(self, dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    
    def split_Train_validation(self, data, column, val_size, look_back):
            df = data[column].values.reshape(-1,1)
            df = self.MinMaxScaler.fit_transform(df)
            
            train = df
            #split into train and validation sets
            validation_size = int(train.shape[0] * val_size)
            train_finalSize = len(train) - validation_size
            validation, train_final = train[0:validation_size,:],  train
            
            print('size of train data:', train_final.shape)
            print('size of validtion data:', validation.shape)
            
            # reshape into X=t and Y=t+1
            trainX, trainY = self.create_dataset(train_final, look_back)
            validationX, validationY = self.create_dataset(validation, look_back)
            
            print('reshape input to be [samples, time steps, features]')
            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            validationX = np.reshape(validationX, (validationX.shape[0], 1, validationX.shape[1]))
            
            print(f'Training shape: {trainX.shape}')
            print(f'Validation shape: {validationX.shape}')

            return(df, trainX, trainY, validationX, validationY)
    
    def prepare_test(self, data, column, look_back):
        df = data[column].values.reshape(-1,1)
        df = self.MinMaxScaler.fit_transform(df)
            
        test = df
        testX, testY = self.create_dataset(test, look_back)
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
        print(f'Testing shape: {testX.shape}')
        
        return(df, testX, testY)
        
        
    def root_cause(self,trainX, trainY, validationX, validationY, neuron_size_first_layer, 
                   neuron_size_second_layer, activation_1, activation_2,
                   look_back, loss, optimizer, epochs_num, batch_size):
        
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(neuron_size_first_layer, activation=activation_1, return_sequences=True, input_shape=(1, look_back)))
        model.add(LSTM(neuron_size_second_layer, activation=activation_2))
        model.add(Dense(1))
        model.compile(loss=loss, optimizer=optimizer)

        # Early stopping
        early_stopping = callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        history = model.fit(trainX, trainY, epochs=epochs_num, batch_size=batch_size, verbose=1,
                            validation_data=(validationX, validationY), callbacks=[early_stopping])
        
        return(history, model)
    
    def compute_score(self, model, X, Y):
        # make predictions
        Predict = model.predict(X)
        # invert predictions
        Predict = self.MinMaxScaler.inverse_transform(Predict)
        Y = self.MinMaxScaler.inverse_transform([Y])
        # calculate root mean squared error
        Score = np.sqrt(mean_squared_error(Y[0], Predict[:,0]))
        print('Train Score: %.2f RMSE' % (Score))
    
        return Predict
    
    
    def create_df_with_prediction(self, data, look_back, Predict, Time, type):
        PredictPlot = np.empty_like(data)
        PredictPlot[:, :] = np.nan
        PredictPlot[look_back:len(Predict)+look_back, :] = Predict
        
        a = list(self.MinMaxScaler.inverse_transform(data).reshape(1, -1)[0])
        b = list(PredictPlot.reshape(1, -1)[0])
        
        df = pd.DataFrame({
                            'time' : Time,
                            'orginal_data' : a, 
                            f'predict_{type}' : b,
                        })
        
        return df

