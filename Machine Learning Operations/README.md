
# Project 2: Operationalizing Machine Learning

## Overview
This project focused on exploration of machine learning model deployments and employed best practices to ensure security and robustness of ML model endpoints. I used a public bank marketing data set that is described on the UCI ML Repo website" "The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed." (https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

By training models on this data set using AutoML the modeling work was taken care of and I could focus on deploying the best model to a working endpoint. I did all of this through the Azure ML UI which proved out the pipeline so I could then automate the pipeline in code and deploy the whole thing as a single endpoint.  

## Architecture Diagram

(TBD)


## Walkthrough

### Step 1: Load Data

#### Load Bank Marketing Datatset
I was able to load in the cleaned bank market data set from the internet and run it through AutoML to begin.
<img width="659" alt="dataset" src="https://user-images.githubusercontent.com/19579908/125129940-64d42e80-e0bd-11eb-949c-5d2df962fef8.PNG">


### Step 2: Train a Model

#### AutoML Experiment Run
This is the completed experiment:
<img width="668" alt="experiment_complete" src="https://user-images.githubusercontent.com/19579908/125303862-5ae34300-e2ea-11eb-8804-674e3803f343.PNG">

#### Best Model
<img width="763" alt="model" src="https://user-images.githubusercontent.com/19579908/125130051-8f25ec00-e0bd-11eb-8247-f409b08d02bb.PNG">



### Step 3: Deploy Model
Now that AutoML had evaluated several models the best model was chosen and deployed to an Azure Container Instance that required authentication.

#### Deployed Model Details
<img width="683" alt="endpoint" src="https://user-images.githubusercontent.com/19579908/125130209-dca25900-e0bd-11eb-8d3f-c91b8df1947f.PNG">

#### Turn on Application Insights
This endpoint then had application insights turned on so that appropriate endpoint logging could occur to alert me should the endpoint fail.
<img width="452" alt="logs" src="https://user-images.githubusercontent.com/19579908/125304299-b44b7200-e2ea-11eb-996b-a82d9ebd4bdc.PNG">
Here is an example of what is logged by default:
<img width="396" alt="app_insights" src="https://user-images.githubusercontent.com/19579908/125130530-618d7280-e0be-11eb-875a-c271c54b1bdb.PNG">

#### Test Deployed Model Endpoint
First in Swagger:
<img width="806" alt="bankmarketing_swagger_ui_endpoint" src="https://user-images.githubusercontent.com/19579908/125319819-5756b880-e2f8-11eb-85dc-9145226351a4.PNG">

Then to test if the endpoint was working test data was sent to the endpoint which returned with this output confirming it worked 
<img width="289" alt="endpoint_success" src="https://user-images.githubusercontent.com/19579908/125130276-f643a080-e0bd-11eb-98d8-ccdad3566b20.PNG">


### Step 4: Automate Pipeline
Unfortunately a lot of this was done in the Azure ML UI which means it's not very scalable or automated. To better automate this process I created a pipeline endpoint using the Azure ML SDK. I also created a pipeline to feed the endpoint. What resulted was a pipeline of steps that takes in new cleaned data, trains a model with it, and deploys the model. 

#### Pipeline Runs
The pipeline creation can be seen here in the pipeline and experiment tabs:
<img width="777" alt="pipelines" src="https://user-images.githubusercontent.com/19579908/125311745-e1028800-e2f0-11eb-9a63-d48be4342de6.PNG">
<img width="566" alt="pipeline_experiment" src="https://user-images.githubusercontent.com/19579908/125314891-d85f8100-e2f3-11eb-9202-52f5ed1fb3dc.PNG">


#### Pipeline API Endpoint
Then the pipeline was published and the endpoint details can be found here:
<img width="344" alt="pipeline_REST_endpoint" src="https://user-images.githubusercontent.com/19579908/125131152-80403900-e0bf-11eb-8dab-b9f50055ab32.PNG">
<img width="207" alt="pipeline_designer" src="https://user-images.githubusercontent.com/19579908/125311874-fe375680-e2f0-11eb-9857-0dcdd7daeb6f.PNG">
<img width="496" alt="endpoint_published" src="https://user-images.githubusercontent.com/19579908/125312644-b402a500-e2f1-11eb-8ccf-df177d0413ab.PNG">


## Screencast
A quick visual walkthrough of the working endpoints can be seen here: https://youtu.be/Ns1OKwm_YyE

## Improvements
If I were to spend more time on this project and not merely prove out a concept I would set up model performance reporting around it. I'd be curious how the model performed over time and would want to see if data drift or model drift creeps in. I'd also spend more time doing custom modeling beyond what AutoML did to see if the model is optimal before deploying it.
