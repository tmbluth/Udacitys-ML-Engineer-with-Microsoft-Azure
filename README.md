# Udacitys-ML-Engineer-with-Microsoft-Azure

## Project 2: Operationalizing Machine Learning

### Overview

This project focused on exploration of machine learning model deployments and employed best practices to ensure security and robustness of ML model endpoints. By training and deploying models to a working endpoint through the UI I was then confident enough to automate that pipeline in code and deploy the whole pipeline as a single endpoint. Then when data was sent to it a model could be trained and deployed automatically.  


### Walkthrough

I was able to load in a cleaned bank market data set from the internet and run it through AutoML to begin.

<img width="659" alt="dataset" src="https://user-images.githubusercontent.com/19579908/125129940-64d42e80-e0bd-11eb-949c-5d2df962fef8.PNG">

<img width="763" alt="model" src="https://user-images.githubusercontent.com/19579908/125130051-8f25ec00-e0bd-11eb-8247-f409b08d02bb.PNG">

<img width="787" alt="automl_run" src="https://user-images.githubusercontent.com/19579908/125129995-7b7a8580-e0bd-11eb-8b31-1b2edd3face2.PNG">

Now that AutoML had evaluated several models the best model was chosen and deployed to an Azure Container Instance that required authentication.

<img width="683" alt="endpoint" src="https://user-images.githubusercontent.com/19579908/125130209-dca25900-e0bd-11eb-8d3f-c91b8df1947f.PNG">

This endpoint then had application insights turned on so that appropriate endpoint logging could occur to alert me should the endpoint fail. Here is an example of what is logged by default

<img width="396" alt="app_insights" src="https://user-images.githubusercontent.com/19579908/125130530-618d7280-e0be-11eb-875a-c271c54b1bdb.PNG">

To test if the endpoint was working example test data was sent to the endpoint which returned with this output confirming it worked 

<img width="289" alt="endpoint_success" src="https://user-images.githubusercontent.com/19579908/125130276-f643a080-e0bd-11eb-98d8-ccdad3566b20.PNG">

Unfortunately a lot of this was done in the Azure ML UI which means it's not very scalable or automated. To better automate this process I created the same pipeline using the Azure ML SDK. 

What resulted was a different endpoint but not one that only took a JSON payload to give back a score. This one was a package of steps that takes in new cleaned data, trains a model with it, and publishes the model to an endpoint.

The final result being this: 
<img width="344" alt="pipeline_REST_endpoint" src="https://user-images.githubusercontent.com/19579908/125131152-80403900-e0bf-11eb-8dab-b9f50055ab32.PNG">


### Screencast

A quick visual walkthrough of the working endpoints can be seen here: https://youtu.be/Ns1OKwm_YyE

### Improvements

If I were to spend more time on this project and not merely prove out a concept I would set up model performance reporting around it. I'd be curious how the model performed over time and would want to see if data drift or model drift creeps in. I'd also spend more time doing custom modeling beyond what AutoML did to see if the model is optimal before deploying it.

