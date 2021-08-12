import argparse
import json
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import Callback
from azureml.core import Run
from azureml.core import Dataset
from azureml.core.workspace import Workspace
import tensorflow
print('\nTensorflow version:', tensorflow.__version__)

def dnn_prep(df):
    target_col = ['Close_ETH']
    df = df[[c for c in df if c not in target_col] + target_col]
    values = df.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    X, y = scaled[:, :-1], scaled[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X, y, scaler

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['val_loss'])

# -------------------------------------------------------------------

run = Run.get_context()

# Add arguments to script
parser = argparse.ArgumentParser()

parser.add_argument('--input_data', type=str)
parser.add_argument('--hidden', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--dropout', type=float)

args = parser.parse_args()
input_data = args.input_data

# Use current workspace 
ws = run.experiment.workspace

# get the input dataset by ID
dataset = Dataset.get_by_id(ws, id=args.input_data)

# load the TabularDataset to pandas DataFrame
crypto = dataset.to_pandas_dataframe()

crypto.set_index('Date', inplace=True, drop=True)
crypto.sort_index(ascending=True, inplace=True)
crypto.head()

target_col = ['Close_ETH']

# Train set
train_size = int(round(crypto.shape[0]*0.8, 0))
train_df = crypto.iloc[0:train_size,:] # this is what makes this df unique
train_df = train_df[[c for c in train_df if c not in target_col] + target_col]
train_values = train_df.values.astype('float32')
### Use scaler in training, validation, and test sets
scaler = MinMaxScaler(feature_range=(0, 1)) 
### Notice `fit_transform` is used only here. For other data sets only `transform` is used
train_scaled = scaler.fit_transform(train_values)
X_train, y_train = train_scaled[:, :-1], train_scaled[:, -1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print('Train:', X_train.shape, y_train.shape)

# Validation set
val_size = int(round(crypto.shape[0]*0.1, 0))
val_df = crypto.iloc[train_size:(train_size + val_size),:] # this is what makes this df unique
val_df = val_df[[c for c in val_df if c not in target_col] + target_col]
val_values = val_df.values.astype('float32')
val_scaled = scaler.transform(val_values)
X_val, y_val = val_scaled[:, :-1], val_scaled[:, -1]
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
print('Validation:', X_val.shape, y_val.shape)

# Test set
test_df = crypto.iloc[(train_size + val_size):crypto.shape[0],:] # this is what makes this df unique
test_df = test_df[[c for c in test_df if c not in target_col] + target_col]
test_values = test_df.values.astype('float32')
test_scaled = scaler.transform(test_values)
X_test, y_test = test_scaled[:, :-1], test_scaled[:, -1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print('Test:', X_test.shape, y_test.shape)

run.log("Hidden Layers", np.int(args.hidden))
run.log('Learning Rate', np.float(args.learning_rate))
run.log("Dropout", np.float(args.dropout))

# Build an LSTM model
model = Sequential()
model.add(LSTM(args.hidden, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(args.dropout))
model.add(LSTM(np.int(args.hidden/2), return_sequences=False))
model.add(Dropout(args.dropout))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError(), 'mae', 'mape'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=25,
    verbose=2,
    validation_data=(X_val, y_val),
    callbacks=[LogRunMetrics()]
)

print(history)

score = model.evaluate(X_val, y_val, verbose=0)
rmse = score[0]
print('Root Mean Squared Error:', rmse)
run.log('Root Mean Squared Error', np.float(rmse))
mae = score[1]
print('Mean Absolute Error:', mae)
run.log('Mean Absolute Error', np.float(mae))
mape = score[2]
print('Mean Absolute Percentage Error:', mape)
run.log('Mean Absolute Percentage Error', np.float(mape))

plt.figure(figsize=(6, 3))
plt.title('ETH RMSE: Train vs Validation Data', fontsize=14)
plt.plot(history.history['root_mean_squared_error'], 'b--', label='Train RMSE', lw=4, alpha=0.5)
plt.plot(history.history['val_root_mean_squared_error'], 'r--', label='Val RMSE', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
run.log_image('ETH_keras', plot=plt)

# create a ./outputs folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs', exist_ok=True)

joblib.dump(scaler, './outputs/scaler.joblib')

np.save('./outputs/X_train.npy', X_train)
np.save('./outputs/y_train.npy', y_train)

np.save('./outputs/X_test.npy', X_test)
np.save('./outputs/y_test.npy', y_test)

model.save('./outputs/ETH_hyperdrive_model') # new way to save and load models
model.save('./outputs/ETH_hyperdrive_model.h5') # old way using hdf5




# # serialize NN architecture to JSON
# model_json = model.to_json()
# # save model JSON
# with open('./outputs/model.json', 'w') as f:
#     f.write(model_json)
# # save model weights
# model.save_weights('./outputs/ETH_hyperdrive_model.h5')
print("model saved in ./outputs folder")



