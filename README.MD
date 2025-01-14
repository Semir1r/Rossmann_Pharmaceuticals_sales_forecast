Customer Purchasing Behavior & Store Sales Prediction
This project explores customer purchasing behavior across various stores and uses machine learning techniques to predict store sales up to 6 weeks ahead. The goal is to analyze factors such as promotions, seasonal behavior, and competitor distance to understand purchasing patterns, and use this knowledge to predict future sales.

Table of Contents
Overview
Task 1 - Exploration of Customer Purchasing Behavior
Task 2 - Prediction of Store Sales
2.1 Preprocessing
2.2 Building Models with Sklearn Pipelines
2.3 Choose a Loss Function
2.4 Post Prediction Analysis
2.5 Serialize Models
2.6 Building Model with Deep Learning
Task 3 - Model Serving API Call
3.1 Framework Selection
3.2 Loading the Model
3.3 API Endpoints
3.4 Deployment
Installation
Usage
Contributing
License
Overview
This project consists of three main tasks:

Exploring customer purchasing behavior using exploratory data analysis (EDA), followed by cleaning the data to handle outliers and missing values.
Predicting store sales using machine learning models to forecast sales for up to 6 weeks ahead.
Serving the trained models via an API, so that predictions can be made in real-time.
Task 1 - Exploration of Customer Purchasing Behavior
This task involves performing exploratory data analysis (EDA) to uncover insights into customer purchasing behavior. The primary goals include:

Checking for distribution differences between the training and test sets
Analyzing sales behavior around holidays, seasonal events, and store openings/closings
Investigating the impact of promotions on sales and customer behavior
Understanding the effect of competitor proximity on sales
Questions Explored:
How are promotions distributed in the training vs. test sets?
What is the sales behavior before, during, and after holidays?
How do seasonal events like Christmas and Easter affect sales?
What is the correlation between sales and the number of customers?
How do promos affect sales and customer attraction?
Task 2 - Prediction of Store Sales
2.1 Preprocessing
Handle missing data and convert categorical columns to numeric.
Extract features such as weekdays, weekends, and holidays from datetime columns.
Scale the data using standard scalers to normalize numerical features.
2.2 Building Models with Sklearn Pipelines
Build machine learning models using pipelines to ensure modularity and reproducibility.
Start with Random Forest Regressor to predict store sales, and evaluate feature importance.
2.3 Choose a Loss Function
Select a loss function that is suitable for regression tasks and justifies its use based on the problem.
2.4 Post Prediction Analysis
Explore feature importance and perform a confidence interval analysis for predictions.
2.5 Serialize Models
Serialize the trained models using timestamps, ensuring each version of the model can be tracked over time.
2.6 Building Model with Deep Learning
Build a Long Short Term Memory (LSTM) model for sales prediction.
Check whether the time series data is stationary, handle autocorrelation, and create a sliding window for time series data.
Transform the data into a supervised learning format and scale it before feeding it into the LSTM.
Task 3 - Model Serving API Call
3.1 Framework Selection
Choose a suitable framework like Flask, FastAPI, or Django REST framework for creating the API endpoints.
3.2 Loading the Model
Load the serialized models for making real-time predictions.
3.3 API Endpoints
Create endpoints to receive customer/store data and return predicted sales.
3.4 Deployment
Deploy the API to a cloud platform or web server to make the model accessible for real-time use.
Installation
Requirements
Ensure you have the following installed:

Python 3.7+
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, tensorflow, flask, fastapi, etc.
For deep learning: tensorflow or pytorch
Install the dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Data Exploration and Cleaning:

Run the explore_data.py to explore and clean the dataset.
Model Training:

Execute the train_models.py script to preprocess the data, train machine learning models, and serialize them.
API Deployment:

Deploy the trained model using app.py to serve predictions via an API.
Making Predictions:

Send POST requests to the API with the necessary data, and receive sales predictions.
Contributing
Feel free to fork the repository, submit issues, and make pull requests. If you have any questions, open an issue and we'll be happy to help.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Author
This project was developed by Semir1r.