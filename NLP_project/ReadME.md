# Natural Language Proecssing Project
## Libraries
+ NumPy
+ Pandas
+ Seaborn
+ Matplotlib
+ folium
+ nltk
+ sklearn

## Project Motivation
For this project I was interested in conducting exploratory data analysis using a Boston Airbnb dataset found on Kaggle containing approximately one year recording and reviews to better understand:

1. When are the busiest times of the year to visit Boston? And how much do price spike?
2. Does the properties in different regions of Boston have influence on the price? and how are the properties distributed on Boston city?
3. Does the superhost get higher review points?
4. What are the features that influence the review points rating? Could those features be utilized to predict review score rating?

## File Descriptions
There are three csv files containing the original information [Kaggle](https://www.kaggle.com/airbnb/boston)
+ calendar, including listing id and the price and availability for that day
+ listing, including full descriptions and average review score
+ reviews, including unique id for each reviewer and detailed comments

One exploratory Jupyter [notebook](https://github.com/langyunlongxmen/DS_project/blob/main/Project_1/CRISP_DM_Boston_Airbnb.ipynb) available here to show what I found: 
1. The september and October are the buiest time in Boston Airbnb market, also will raise the prices.
2. Jamaica Plain, South End, Back Bay, Fenway, Dorchester are the TOP 5 regions with propertier distribution.
3. The superhosts have 6.6% higher review points.
4. Utilizing information about location, host, room, price to predict review_score with 0.08791 r2_error for testing data.
 
## Acknowledgements
Acknowledgement should go to Kaggle for providing the dataset. 
