# Disaster Response ETL, ML pipelines and Web application.
## Project Motivation
The purpose of the project is to build a model for an API that classifies disaster messages. Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed: "water", "shelter", "food", etc.

The web app also displays visualizations of the data.

## Web application screenshots
![file1](https://github.com/langyunlongxmen/DS_project/blob/main/Project_2/pic%20(1).png)

file2

file3

## Install
This project requires Python 3.x and the following Python libraries installed:

NumPy
Pandas
Matplotlib
Json
Plotly
Nltk
Flask
Sklearn
Sqlalchemy
Sys
Re
Pickle
You will also need to have software installed to run and execute an iPython Notebook

## File Descriptions
process_data.py: This code extracts data from both CSV files: messages.csv (containing message data) and categories.csv (classes of messages) and creates an SQLite database containing a merged and cleaned version of this data.
train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
ETL Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py automates this notebook.
ML Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which model to use. train_classifier.py automates the model fitting process contained in this notebook.
disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.
templates folder: This folder contains all of the files necessary to run and render the web app.
custom_transformer.py contains custom functions that were used in ML Pipeline Preparation.ipynb so to find best way of model tuning.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
Run the web application Go to http://0.0.0.0:3000/ (if facing problems try http://localhost:3000 in a browser)

In the web app you may input any text message (English) and it will categorize it among 35 classes.

## Data observations
As can be seen from test data visualization most of the classes (categories) are highly imbalanced. This affects model F1 prediction score. One using this project should take this into consideration and apply measures like synthetic data generation, model selection and parameters fine-tuning, etc.

## License
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.
