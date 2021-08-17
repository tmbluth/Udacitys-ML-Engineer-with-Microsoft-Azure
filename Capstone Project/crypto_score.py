
import joblib
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from azureml.core.model import Model

def init():
    global model

    try:
        model_dir = Model.get_model_path('hyperdrive_Ethereum_forecaster')
        print(model_dir)
        print(os.listdir(model_dir))
        model = load_model(os.path.join(model_dir, 'ETH_hyperdrive_model'))
    except Exception as e:
        return 'Hyperdrive model loading failed due to following error:\n' + str(e)
        print()

    
def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        # Make prediction
        y_hat = model.predict(data)
        y_hat_list = y_hat.tolist()
        return y_hat_list 
    except Exception as e:
        return str(e)
