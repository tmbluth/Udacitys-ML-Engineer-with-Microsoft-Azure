import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model

def init():
    global model

    ### KERAS APPROACH 1 --------------------------------------------
    # if os.path.exists(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs')):
    #     model_dir = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs')
    # else:
    #     model_dir = os.getenv('AZUREML_MODEL_DIR')
    # print(model_dir)

    # model = load_model(os.path.join(model_dir, 'ETH_hyperdrive_model'))

    ### KERAS APPROACH 2 --------------------------------------------
    # from azureml.core.authentication import ServicePrincipalAuthentication
    # from azureml.core.model import Model
    # from azureml.core.workspace import Workspace

    # svc_pr_password = os.environ.get("AZUREML_PASSWORD")
    # tenant_id = os.environ.get('tenant_id')
    # subscription_id = os.environ.get('subscription_id')
    # resource_group = os.environ.get('resource_group')
    # # what are these?
    # client_id = os.environ.get('client_id') 
    # client_secret = os.environ.get('client_secret')

    # svc_pr = ServicePrincipalAuthentication(
    #     tenant_id=tenant_id,
    #     service_principal_id="my-application-id???",
    #     service_principal_password=svc_pr_password)

    # ws = Workspace(
    #     subscription_id=subscription_id,
    #     resource_group=resource_group,
    #     workspace_name="ml-devtest",
    #     auth=svc_pr
    #     )
    # model_dir = Model.get_model_path('hyperdrive_Ethereum_price_forecast', _workspace=ws)
    # print(model_dir)
    # print(os.listdir(model_dir))
    # model = load_model(os.path.join(model_dir, 'ETH_hyperdrive_model'))

    ### AUTOML APPROACH 1 --------------------------------------------
    # print(os.listdir(os.getenv('AZUREML_MODEL_DIR')))
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_Ethereum_price_forecast', '6', 'model.pkl')
    # print(model_path)
    # model = joblib.load(model_path)

    # # AUTOML APPROACH 2 --------------------------------------------
    # model_path = Model.get_model_path('automl_Ethereum_price_forecast')
    # model = joblib.load(model_path)

    ### BOTH MODEL TYPES AND APPROACHES ------------------------------
    try:
        print(os.listdir(os.getenv('AZUREML_MODEL_DIR')))
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_Ethereum_price_forecast', '6', 'model.pkl')
        print(model_path)
        automl_model = joblib.load(model_path)
    except:
        print('AutoML model loading failed')

    try:
        model_dir = Model.get_model_path('hyperdrive_Ethereum_price_forecast')
        print(model_dir)
        print(os.listdir(model_dir))
        hyperdrive_model = load_model(os.path.join(model_dir, 'ETH_hyperdrive_model'))
    except:
        print('Hyperdrive model loading failed')


    
def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # make prediction
    y_hat = hyperdrive_model.predict(data)
    return y_hat.tolist()