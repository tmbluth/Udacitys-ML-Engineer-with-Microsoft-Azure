
# Project 2: Operationalizing Machine Learning

## Overview

This project focused on exploration of machine learning model deployments and employed best practices to ensure security and robustness of ML model endpoints. I used a public bank marketing data set that is described on the UCI ML Repo website" "The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed." (https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

By training models on this data set using AutoML the modeling work was taken care of and I could focus on deploying the best model to a working endpoint. I did all of this through the Azure ML UI which proved out the pipeline so I could then automate the pipeline in code and deploy the whole thing as a single endpoint.  

## Architecture Diagram

![AML Engineer Udacity Project 2 Diagram](https://user-images.githubusercontent.com/19579908/125333652-ba038080-e307-11eb-8283-dcfc1945c305.png)



## Walkthrough

### Step 1: Load Data

#### Load Bank Marketing Datatset

I first loaded in cleaned bank market data set from the internet and to be run through AutoML for starters.

<img width="659" alt="dataset" src="https://user-images.githubusercontent.com/19579908/125129940-64d42e80-e0bd-11eb-949c-5d2df962fef8.PNG">



### Step 2: Train a Model

#### AutoML Experiment Run

I ran the AutoML experiment with a time constraint of 1 hour to speed up the process. All I needed to demonstrate the deployment concepts was a working model, not an optimal one. The completed experiment can be seen here:

<img width="668" alt="experiment_complete" src="https://user-images.githubusercontent.com/19579908/125303862-5ae34300-e2ea-11eb-8804-674e3803f343.PNG">

#### Best Model

I then clicked the best model to view its details and deploy it.

<img width="763" alt="model" src="https://user-images.githubusercontent.com/19579908/125130051-8f25ec00-e0bd-11eb-8247-f409b08d02bb.PNG">



### Step 3: Deploy Model

Once I clicked the "Deploy" button on the best model I had to configure the deployment. I chose to create an Azure Container Instance that required authentication and sent it off.

#### Deployed Model Details

Once the deployment completed it looked like this:

<img width="683" alt="endpoint" src="https://user-images.githubusercontent.com/19579908/125130209-dca25900-e0bd-11eb-8d3f-c91b8df1947f.PNG">

You can see the URI endpoint and the swagger URI that provides data to make interaction with the API endpoint more structured. For example it provides expected input names, data types, and standardized messages among other things.

#### Turn on Application Insights

This endpoint started with Application Insights off but then had it turned on through the logs.py file.

<img width="452" alt="logs" src="https://user-images.githubusercontent.com/19579908/125304299-b44b7200-e2ea-11eb-996b-a82d9ebd4bdc.PNG">

Here is an example of what is logged by default:

<img width="396" alt="app_insights" src="https://user-images.githubusercontent.com/19579908/125130530-618d7280-e0be-11eb-875a-c271c54b1bdb.PNG">

This piece is crucial because a data scientist might be tempted to just put a model out into production and only track model performance, but not endpoint performance. We need a way to track our endpoints' health so that we can fix it when needed for optimize it where necessary. Knowing failure rates, response time, and requests can help us do just that.

#### Test Deployed Model Endpoint

This can first be tested in Swagger. By running the swagger.sh file to run a Docker container with Swagger I was then able to run serve.py to create a webserver to send data through. The Swagger UI is a place we can check that the endpoint is active and receiving requests and returning responses.

<img width="806" alt="bankmarketing_swagger_ui_endpoint" src="https://user-images.githubusercontent.com/19579908/125319819-5756b880-e2f8-11eb-85dc-9145226351a4.PNG">

For this use case I needed to send a JSON payload through my HTTP webserver so I ran endpoint.py with example data to do so. If the endpoint is working the test data sent would return back output responses. 

<img width="170" alt="endpoint_success" src="https://user-images.githubusercontent.com/19579908/125344657-cf32dc00-e314-11eb-883d-56a0c10c7bed.PNG">

The test data works! Our endpoint and model are working. Now to take things to the next level.

### Step 4: Automate Pipeline

Unfortunately a lot of this was done in the Azure ML UI which means it's not very scalable or automated. To better automate this process I created a pipeline experiment using the Azure ML SDK and published the pipeline to an endpoint to be consumed. This was done in the aml-pipelines-with-automated-machine-learning-step.ipynb notebook.

#### Pipeline Run

The pipeline creation can be seen in the notebook and the "Pipeline" and "Experiment" tabs:

<img width="344" alt="pipeline_REST_endpoint" src="https://user-images.githubusercontent.com/19579908/125131152-80403900-e0bf-11eb-8dab-b9f50055ab32.PNG">

<img width="777" alt="pipelines" src="https://user-images.githubusercontent.com/19579908/125311745-e1028800-e2f0-11eb-9a63-d48be4342de6.PNG">

<img width="566" alt="pipeline_experiment" src="https://user-images.githubusercontent.com/19579908/125314891-d85f8100-e2f3-11eb-9202-52f5ed1fb3dc.PNG">

Also, when you click into the pipeline run you can see it in the "Designer"

<img width="207" alt="pipeline_designer" src="https://user-images.githubusercontent.com/19579908/125311874-fe375680-e2f0-11eb-9857-0dcdd7daeb6f.PNG">

#### Pipeline API Endpoint

Then the pipeline was published via code and the endpoint details can be found here:

<img width="750" alt="published_pipeline_notebook" src="https://user-images.githubusercontent.com/19579908/125458006-c36c3e8b-3722-41f5-983e-20c841e076e7.PNG">

<img width="496" alt="published_pipeline_details" src="https://user-images.githubusercontent.com/19579908/125312644-b402a500-e2f1-11eb-8ccf-df177d0413ab.PNG">

You can see the pipeline is published and active. I then tested it with the endpoint.py file and it worked just like the manual pipeline we created above.


## Screencast

A quick visual walkthrough of the working endpoints can be seen here: https://vimeo.com/574131393/d3b46d70a1



## Improvements

If I were to spend more time on this project and not merely prove out a concept I would set up model performance reporting around it. I'd be curious how the model performed over time and would want to see if data drift or model drift creeps in. I'd also spend more time doing custom modeling beyond what AutoML did to see if the model is optimal before deploying it. Most importantly I would tailor this project to fit into the software architecture we've established at work to prove out these concepts in that environment so we might take the best practices learned here and use them there.
