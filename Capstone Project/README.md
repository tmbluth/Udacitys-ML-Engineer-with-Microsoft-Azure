# Ethereum Forecaster

This year was a very exciting time for cryptocurrency. In late 2020 it began to boom once more and reached historical highs in the spring of 2021. During the middle part of the boom I jumped into the market and have learned a lot about how cryptocurrency works from a technology perspective and from a trading perspective. In this project I hope to take some of my learnings and see what might happen if I had applied them in the past.

Given the current economy of cryptocurrency is largely driven by Bitcoin I am using it as a leading signal for the price of alternative coins. In this case, specifically one of my current favorites, Ethereum. 



## Project Set Up and Installation
To complete this project I used a work subscription to Azure which had Azure Blob Storage set up already. Before being able to run this code yourself you will need to have an Azure subscription and blob storage established. You will also need to download the data using the Kaggle API which I walk though in my notebooks. Just to be clear, the main files in this project are `automl.ipynb` and `hyperparameter_tuning.ipynb`. The rest of the files are created from within these 2 files as you will see when you walk through them.

## Dataset

### Overview
The data sets I'll be working with are from Kaggle and contain time-series price data on popular cryptocurrencies from their Initial Coin Offering (ICO) to July 2021. 

If this is true, using daily Bitcoin data should be a good signal for Ethereum's daily closing price. 

### Task
For this project we will only be using historical info from Bitcoin (BTC) and Ethereum (ETH) to predict the closing price of Ethereum at each day's open. BTC has been known to be the market leader with a huge portion of market capitalization and tends to heavily influence the prices of alternative coins like ETH. Features like coin trading volume, market capitalization, highs, lows, moving averages, and standard deviations are all part of the time series inputs to our models.

### Access
When I log in for the first time I use the default interactive authentication for this version of the AzureML SDK upon requesting access to the default blob storage. It prompts the user to input a code provided from the notebook into a separate authentication page.

## Automated ML
The settings and configs for the AutoML runs were chosen to balance training speed and model accuracy. 

Training Speed:
I chose the run time to not exceed 4 hours by changing `experiment_timeout_hours` to ensure some time to play with the resulting models the same day I run this code. To minimize the time it takes to do the training I blocked a few modeling algorithms with `blocked_models` since I know they generally don't perform well on tasks like this. I also bumped up the computation to be spread across 5 of the 6 possible nodes with `max_concurrent_iterations` which leaves one for general purpose programming and orchestration as needed. Finally I turned on `enable_early_stopping` so that training would cease if a model was stagnant or worse as a result of continued training iterations.

Model Accuracy:
I chose a `normalized_root_mean_squared_error` (normalized RMSE) as the primary evaluation metric for the models to compete on. Regular RMSE gives an accurate representation of how far off the predictions are in either direction from the true price. That value is then normalized to account for any scaling discrepancies between modeling algorithms. This normalization will allow AutoML to compare all models apples to apples and pick the best model. This metric was calculated setting `validation_data` to our validation dataset to evaluate out-of-time data instead of random time periods within the trainig dataset. I also turned on `featurization` to `auto` which takes longer but could be important in automatically creating training features that would boost model performance.

### Results

The range in performance varied pretty widely but the best model had performance of 5.64% MAPE with the validation set. The best model was a VotingEnsemble made up of 8 different models that were elastic nets and tree based algorithms. I could've improved this model by finding more rich features, but overall I think it did pretty well. Here is a screenshot of the completed run and the ensemble details.

<img width="458" alt="AutoMLPipelineRunDetail" src="https://user-images.githubusercontent.com/19579908/129660168-dba0cf2e-097c-4a6f-82c0-661845f550e9.PNG">

<img width="497" alt="BestAutoMLModelRun" src="https://user-images.githubusercontent.com/19579908/129766277-e9a5b8ed-87bf-4f41-8493-30a9fcaf2c7d.PNG">

<img width="314" alt="AutoMLVotingEnsemble" src="https://user-images.githubusercontent.com/19579908/129660184-c0110e5b-e236-4d3c-af54-d74f5228d177.PNG">

I registered the model from the PipelineRun with the following code:

```
best_model_output_name='best_model_output'
best_model_output = run.get_pipeline_output(best_model_output_name)
best_model_output.download('./automl_model', show_progress=True)

model_path = './automl_model/' + str(best_model_output._path_on_datastore)
with open(model_path, "rb" ) as f:
    best_model = joblib.load(f)

run.upload_file(name=model_path, path_or_stream=model_path)
run.register_model(
    model_name = 'automl_Ethereum_forecaster', 
    model_path = model_path,
    description = 'Auto ML training',
    tags=None)
```

## Hyperparameter Tuning
Instead of traditional forecasting methods I decided to use a multivariable LSTM approach that predicted one time series ahead. This was done since LSTM is a leading algorithm in forecasting problems. The parameters I optimized were dropout, learning rate, and hidden neuron count. The details of these hyperparameters are found in the `hyperparameter_tuning.ipynb` notebook. 

### Results

This model actually did better than the AutoML model on the validation set with a MAPE of 2.97%. It had a learning rate of 0.072, a dropout of 0.158 on 2 layers and 2 LSTM layers with 32 and 16 neurons respectively. The only problem was the test set performance. Since the last 10% of the data captured a major bull run follow by a crash all models trained had a very hard time predicting the prices. The best HyperDrive model got an 18% MAPE which I personally think is unusable in the real world. This may be fixed by including the volatility that the test set displayed in the next training data sset and collecting more recent tame data for testing.

The best model run and details can be seen below.

<img width="746" alt="HyperDriveRunDetail" src="https://user-images.githubusercontent.com/19579908/129660816-9a6dcc1f-bc6e-4a2c-b15b-9b9c0b601ab2.PNG">

<img width="921" alt="HyperDriveModel" src="https://user-images.githubusercontent.com/19579908/129660899-85257584-b401-4844-9c6e-d4d1c81737e9.PNG">

## Model Deployment
I chose to deploy the hyperparameter model because I wanted experience deploying a deep learning algorithm into production. The deployed model is packaged using Docker and saved in an Azure Container Instance. It was then deployed to a webservice with 2 nodes of compute and 1GB of memory. The environment I used to deploy the endpoint is the same environment I used to train the model to ensure compatibility with model loading and data ingestion. This environment can be recreated by using the `dockerfile` in the repo, which is also created through the notebook.

When the model was trained a MinMaxScaler (scikit-learn) was created on the training set and saved with the model run. If you want to feed the endpoint some data you must use this scaler on the 17 columns of Ethereum data (including the target feature "Close-ETH"), break out the 16 inputs, reshape them to be Keras compatible, and convert them into JSON. You send this data with the `Webservice.run()` method as seen below:

<img width="308" alt="HyperDriveWebserviceRun" src="https://user-images.githubusercontent.com/19579908/129759858-1ec0e884-0421-40e9-99f2-e955289e4910.PNG">

This is the endpoint I hit which should look similar if this code is run in a different workspace:

<img width="486" alt="HyperDriveEndpoint" src="https://user-images.githubusercontent.com/19579908/129760159-a9e3bde1-187e-4d14-bd70-4fafd8785708.PNG">

Once you get a response you can download this scaler and bring the predictions back to their original form with the `inverse_transform()` method of the scaler.


## Screen Recording
https://vimeo.com/588157040/59a742eb0d

## Standout Suggestions
Instead of just deploying the best model to a single endpoint I chose to do a little exploration and prove out many different tools Azure has to offer. The AutoML notebook not only uses AutoML but also creates a batch pipeline that utilizes the AutoML model. In the HyperDrive script I explored deployment of a real-time scoring option by deploying a webservice to be hit at any time, given you have the correct access keys and URL. Both of these approaches cover very different AzureML functionality for the same use case and either can be used.

## Heads Up
If you try to run this code be aware that changing the development environment's Python libraries is extremely difficult and risky. This is because it comes prepackaged to run AutoML. Messing with libraries would make AutoML unusable in many cases. This also made loading Tensorflow models impossible in the workspace since the dev environment was different than the training environment that needed specific versions of Tensorflow to work correctly.
