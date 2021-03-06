# Disaster Response Pipeline Project.
## Project Motivation
The purpose of the project is to build a model to classify disaster messages. Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed: "water", "shelter", "food", etc.

## Web application screenshots
![file1](https://github.com/langyunlongxmen/DS_project/blob/main/Project_2/pic%20(1).png)

![file2](https://github.com/langyunlongxmen/DS_project/blob/main/Project_2/pic%20(2).png)

![file3](https://github.com/langyunlongxmen/DS_project/blob/main/Project_2/pic%20(3).png)

![file4](https://github.com/langyunlongxmen/DS_project/blob/main/Project_2/analyzing%20text.png)

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
process_data.py: This code extracts data from both CSV files: disaster_messages.csv (containing message data) and disaster_categories.csv (categories of messages) and creates an SQLite database containing a merged and cleaned version of this data.
train_classifier.py: This code takes the data of SQLite database as inputs to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. 
ETL Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py development procces.
ML Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which model to use. train_classifier.py process_data.py development procces.


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

## License
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.
