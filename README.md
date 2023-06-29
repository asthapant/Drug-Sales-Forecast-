# Case Study : Drug-Sales-Forecasting

## Analytics in Pharmaceutical Industry

Data analytics plays a crucial role in the pharmaceutical industry, enabling companies to make data-driven decisions, improve drug development processes, enhance patient outcomes, and optimize operational efficiency. This problem statement is focused around the use of machine learning modeling to estimate the potential sales of drugs which are highly impacted by market parameters. 

## Overview
The goal is to develop a robust algorithm to forecast the future sales for a portfolio of two drugs sold in 25 accounts across 5 countries. The drug sales are largely impacted by the account potential, demographics, market factors, company's promotional expenditures and key product launches. The idea is to leverage the historical monthly data pertinent to the year 2011 to 2018, derive insights from it and give possible sales numbers for the year 2019. 

This documentation is to curate my approach followed and methodology adopted in the workflow. 

## Data Description

The data primarily consists of the following files: 

 1. Account level historical sales data for 8 years for the products Drug_1 and Drug_2.
 2. Sales data for a new product launched in the market (Drug_3).
 3. Potential of each account available based on patient size - classified into high, low and medium.
 4. Promotional Expenses for each product in each account and the split numbers into Salesforce and Digital expenses.   

### File Descriptions

    - 02 Sales_Expense_InputFile.csv - Training Data:  Monthly historical data from August 2010 to October 2018.
    - 03 Account_Potential_Data.csv - Account Potential Information, to be merged with historicals for a combined training set
    - final_test_data.csv - Test Data: Sample submission file comprising of accounts list for Drug_1 and Drug_2 sales prediction for the next nine months from November 2018 to July 2019.
    - 
### Data Fields

    - Account_ID - an ID representing a unique account (roughly a store).
    - Type - the potential of each account (high, medium or low)
    - Country - indicating demographics, the country that account is located in (has five countries from Country_1 to Country_5)
    - Product - identifying the product we're referring to (either of Drug_1, Drug_2 or Drug_3)
    - Month - date in format mm/dd/yyyy (always the first of every month).
    - Unit_sales (in million $) - the sale amount for a product in an account. 
    - Per unit expense - the expenditure made for one product unit. 
    - Total_expense (in million $)- total amount spent for a product for promotional reasons. 
    - Salesforce_expense (in million $)- expenditure made in promotion through on ground salesforce deployment.
    - Digital_expense (in million $)- expenditure made in promotion through digital platforms. 
    
## Exploratory Data Analysis

Performed extensive analysis on the dataset to get deeper insights. After checking the missing values and transforming the datatypes of the loaded data, plotted graphs on varied levels to analyze data on diverse fronts. 

Some interesting results obtained were: 

 1. The launch of Drug 3 was on different months for different stores. Its launch did affect the sales of Drug 1 and Drug 2 for some accounts. For instance: Drug 3 was launch in some accounts in January 2014, consequently the sales of Drug 2 dipped down in Jan'14 as compared to its sales in December 2013.  
 2. Drug 2 has considerably higher sales as compared to Drug 1 while Drug 3 has comparatively skewed less sales numbers. 
 3. Country 2,3 and 4 have steeply increasing sales numbers from 2011 to 2018 -while country 1 and 5 witness a dip from 2016. 
4. High potential accounts have recorded the maximum sales with a steady increase upto the year 2017. It has recorded a significant dip of around 16% in its total sales in the year 2018.
5. For all the countries, the salesforce ratio has been majorly above 0.5 for Drug 1 and Drug 2, indicating more on-ground salesforce promotional expenses.
6. For Drug 3, the salesforce ratio has been between 0.35 to 0.5, indicating higher digital expenditure.
7. Drug_1 and Drug_2 yield similar profits across all countries. However, there are high profits for Drug_3 during launch- which dips highly in the subsequent years.
8. All countries record similar profit margins (Country 2 and country 4 do outperform in some cases). But, country 1 records least profits for Drug 3 - indicating scope of potential improvement. 
9. Country 4 has the maximum average sales numbers, followed by Country 3 and Country 5.
10. The average sale numbers for High potential accounts are >100, ranging till 10k million dollars. The medium potential accounts have slightly higher sales, with average at 40 million dollars. Also, profits Made by a High Potential account for a product is higher than that by a low and medium potential account. 

## Feature Engineering

### Generate Additional Features

Date time features are created from the time stamp value of each observation. For my model, I have created features from the monthly date provided for each account. There include : weekoftheyear, quarter, is_month_start, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end features. 

Besides, also added a feature depicting the number of units of product available in an account. This is obtained from the Total Expense and Per Unit Expense columns. 
Added the Salesforce Ratio feature to get the fraction of expenditures made for on-ground promotion through salesforce out of the total. 

### Generate Lag Features

Lag features are the values generated at prior time steps. For my model, I have generated lag features on Total Expenses, Salesforce Expenses and Digital Expenses. Time steps are: 1,2,3,6 and 12 months. The null values generated after the shift are substituted by zeroes.

The lag features of Salesforce Expenses turned out to be one the most important features in the model training, as per gradient boosting’s importance features.

###  Generate Rolling Mean Features

The rolling method is used to derive the moving average feature values. I have used a 3 month rolling window for my dataset and the averages are calculated for three columns - Total Expenses, Salesforce Expenses and Digital Expenses.

### Computing Expense Feature for Months to be Forecasted - An Approach

Since there was no data available on the Expenses made by each account as a part of promotion for each product, I devised and approach to compute the same.

The amount an account will spend to promote a drug can depend on the following two factors : 

 - On the general trend of expenses assigned  in the corresponding year or the past year.
 - On the seasonal monthly variations which occur every year in the targeted months (climate change months can witness surge in sales and thus intuitive increase in promotional expenditures).

1. **To calculate the expenses in the future months, I deployed the following:**

  * Taking the mean of Total Expenses for the year 2018 (Jan-Oct)
  * Taking the value of Expenses for the same month last year.
   * Averaging the two to get the potential expenses made by the company in the particular future month.
   Sample calculation for Total Expenses of November 2018 (whose sales are to be predicted)-
*- Average of (Mean Total Expenses from January to October 2018 + Expenses for November 2017)*

Note: This strategy is deployed to train the model thrice (to get the upper and lower values of predicted sales).

- For the lower value, expenses are :
*- Average of (Mean - Standard Deviation of Total Expenses from January to October 2018 + Expenses for November 2017)*

- For the upper value, expenses are:
 *-Average of (Mean + Standard Deviation of Total Expenses from January to October 2018 + Expenses for November 2017)*

2. **To calculate the number of units in the future months, adopted the same strategy.**

  *Average of (Mean Number of Units from January to October 2018 + Number of Units for November 2017)*

Number of Units are calculated from - Total Expenses/Per Unit Expense

3. **To get the salesforce ratio for future months and hence compute the split of salesforce and digital expenses for the future months:**

Have taken the median salesforce ratio for the year of 2018 (Jan-Oct)

## Cross validations

Since this is time series so we have to be careful with the folds for training and validations. 
Initially, data from 2011 to December 2017 is used for training purposes. Validation data is from January to October 2018. Test data is from November 2018 to July 2019. 

Retrieved the CV indices can be retrieved from the TimeSeriesSplit function during Hyperparameter tuning using Hyperopt: 

    tscv = TimeSeriesSplit(n_splits=15)
    for idx, (train_index, test_index) in  enumerate(tscv.split(x_train)):
    X_train, X_test = x_train.iloc[train_index], x_train.iloc[test_index]
    Y_train, Y_test = y_train.iloc[train_index], y_train.iloc[test_index]

## Training Methodology:

XGBoost is tuned first using GridSearchCV and then using Hyperopt further to get the optimal results. 

The hyperparameters obtained with GridSearchCV yield an RMSE value of 70.90 with R-square value 0.94. Here, I used the K-Fold cross validation to split the data without shuffle. This might not be very accurate as we are dealing with time series information.

Further, I performed Hyperparameter tuning using Hyperopt Library which provided a more refined optimization framework. The RMSE values are also largely affected by min_child_weight hyperparameter. To prevent overfitting, I have also incorporated the lambda_l1 and lambda_l2 (alpha) regularization factors. The learning rate is kept small (0.03) for training.
The objective function is minimized using fmin and Tree-structured Parzen Estimator (tpe) optimization algorithm is used to select the promising hyperparameters. 
 The search space of the tuning is below:
 

    space = {
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 0.9, 0.05),
    'gamma': hp.quniform('gamma', 0, 1, 0.05), 
    'lambda': hp.quniform('lambda', 0, 1, 0.05),
    'alpha': hp.quniform('alpha', 0, 1, 0.05),
    'min_child_weight': hp.quniform('min_child_weight', 5,10, 1)
    'learning_rate': hp.choice('learning_rate',[0.07, 0.03, 0.1]),
    'eval_metric': 'rmse',
    'objective': 'reg:squarederror' ,
    'early_stopping_rounds':100
    }

The hyperparameters obtained with Hyperopt yield an RMSE value of 51.70 with R-square value 0.96.

## Libraries Used

 - python pandas
 - python numpy
 - python matplotlib
 - python seaborn
 - python sklearn
 - python datetime

## Run Steps

The Exploratory Data Analysis notebook can be found here:

[Exploratory Data Analysis](https://github.com/asthapant/Drug-Sales-Forecast-/blob/main/EDA_Notebook.ipynb)

The final model trained and tuned along with generated predicted values can be found here:

[Model Training and Forecast](https://github.com/asthapant/Drug-Sales-Forecast-/blob/main/Model_Training.ipynb)

## Conclusions
The task was to predict the sales of each account for two drugs for a period of next months. We started by exploring the time series data, did some feature engineering to prepare the data for modelling and computed the test data feature values. Finally, we used the  XGBoost model to predict the sales. To improve the model performance, the hyperparameter tuning was done on two levels. This was followed by several instances of trying and testing the parameters to find the most efficient one amongst them. The model takes approximately an hour to tune for both GridSearchCV and Hyperopt. The RMSE value decreased by 20 units post the tuning steps. However, the initial RMSE came out to be more promising. 

## Improvement Areas:
There are several ways in which the model can yield better results: 
- We can use other modeling techniques for time series data like LightGBM, Lasso/RidgeRegression, SARIMAX and take the Ensemble of all as the final forecast.
- Although XGBoost takes care of seasonality and trend but making the data stationary before training the model and then adding the trend and seasonality components later on might have resulted in better results. However this is more pertinent to ARIMA model (the lagged features in our model mitigate the non-stationarity of data).
- Next task is to analyze the Cross Validation folds - if a different method of split can improve model performance.
- The iteration steps in Hyperopt hyperparameter tuning can be increased for reduced RMSE. However this may increase the computation time drastically.  
