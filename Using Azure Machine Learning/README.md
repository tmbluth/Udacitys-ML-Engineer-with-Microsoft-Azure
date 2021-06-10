# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the HyperDrive in the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run using the Python SDK.

## Summary
This dataset contains marketing data from phone marketing campaigns of a bank. The classification goal is to predict whether a client will start an account and make a deposit.

On the measure of accuracy, the best HyperDrive model was a LogisticRegression from scikit-learn and got 0.9137. The inverse of regularization strength (hyperparameter name "C") was 0.1 and the maximum iterations (hyperparameter name "max_iter") was 100. The best AutoML model had an accuracy of 0.9168 by using a VotingEnsemble which slightly beat out custom modeling using HyperDrive.

## Scikit-learn Pipeline
To run the scikit-learn LogisticRegression model I modified a training script that loaded the data, cleaned it, and trained a model using the hyperparameters passed to it. The hyperparameters were specified in the notebook that ran the script and sent it tuning configurations. I chose to search this space of hyperparameters:

    ps = RandomParameterSampling(
      {
        'C': choice(0.001,0.1,1,100,1000),
        'max_iter': choice(100, 250, 500)
      }
    )

This is not an exhaustive search since there were only 10 runs out of 15 possible combinations (5 "C" * 3 "max_iter"). Doing random hyperparameter searching speeds up the process in that we did not have to search all 15 options. 

The early stopping policy I chose was fairly simple and is seen below:

    policy = BanditPolicy(slack_amount=0.1)
    
This checks each iteration of a training algorithm to see if there had been a degredation in model performance. In this case I gave it a "slack" of 0.1, which means if the model iterations deviate negatively from the validation set's evaluation metric by 0.1, then the training will be cut short. Since accuracy was the evaluation metric here, it is pretty unlikely that a training iteration would deviate that much from the best iteration, but it was there just in case some extreme overfitting occurred.

## AutoML
The AutoML model training code was run in the notebook instead of mostly from the training script. There weren't a lot of controls I put on it aside from defining the evaluation metric and training guardrails. I did have to clean the data with a function from the training script and save the cleaned data into a new TabularDataset so that it could be used by AutoML, but other than that, the code was not too complex.

As mentioned above the AutoML run found a VotingEnsemble that beat out all other models. The hyperparameters it landed on were as follows:
    
    {
        "boosting_type": "gbdt",
        "colsample_bytree": 0.6933333333333332,
        "learning_rate": 0.05263631578947369,
        "max_bin": 50,
        "max_depth": 5,
        "min_child_weight": 8,
        "min_data_in_leaf": 0.05517689655172415,
        "min_split_gain": 0,
        "n_estimators": 200,
        "num_leaves": 23,
        "reg_alpha": 0.5789473684210527,
        "reg_lambda": 0.894736842105263,
        "subsample": 0.3963157894736842
    }

## Pipeline comparison
Though the 2 approaches were different the outcome was very similar. The fact that AutoML could compete with our custom solution makes me much more open to using AutoML in the future for simple use cases due to its relative ease of use. Not only is it easy to use in the Python SDK but just about anyone with some business sense can manually run it in the UI for some exploratory work. Having to run a seperate training script for the scikit-learn HyperDrive approach felt clunky and is definitely not preferred if not needed.

However custom tuning has its place. It is important to note that evaluation metrics are not necessarily the only thing we want to optimize. Some algorithms are better suited for things like outlier management than others, and some are optimized for speed. Also, some use cases require model upgrades to have fairly steady behavior and explainibility. If AutoML code is running often and pushing different models using different algorithms into production based on evaluation metrics only, then there is chance for a "better" model to hurt customer operations or strategy. In those cases it would be best to use the same algorithm in every model upgrade to ensure prediction stability.

## Future work
For starters, I'd want to understand the use case better to see if we should be optimizing a different metric. Though accuracy works, use cases vary and other metrics might be more informative about a model's performance. I'd also like to use Bayesian hyperparameter sampling to find the best hyperparameters, though there is potential for that taking longer. Finally I'd choose to run AutoML first with the evaluation metric of choice to find the best algorithm. If that algorithm fit the use case I would optimize it for this use case using HyperDrive. 

## Proof of cluster clean up
Since this cluster is the same one that I use at work and is needed for other proof-of-concept pipelines I chose not to delete it. If I were to delete a cluster I would just add a line of code like this:

    compute_target.delete()


