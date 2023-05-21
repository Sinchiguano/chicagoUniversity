#!/usr/bin/env python
# coding: utf-8

# # Quiz 1: Manipulating Data & Linear  Regressions
# 
# In this quiz, you will get hands-on experience preparing a real-world dataset for modeling. Then, you will applying linear regressions to make predictions. We will use a truncated version of the Divvy Bike Share dataset, which was used in the lab and lecture. 
# 
# After completing this lab, you should be able to: 
# 
# 1. Manipulate a dataset in Python/Pandas/Jupyter Notebooks.
# 2. Implement pre-processing methods on your dataset, including:
#     * Truncating and subseting your data.
#     * Normalizing your dataset in preparation for training and testing.
# 3. Apply the `scikit-learn` Linear Regression model to a real-world dataset. You will be able to:
#     * Split your data into a training and testing set.
#     * Create a model. 
#     * Combine data and metrics from multiple models.
# 4. Evaluate your model using measurements like MAE, MSE and $R^2$.
# 
# We will be working with a truncated version of the [Divvy Trip data](https://data.cityofchicago.org/Transportation/Divvy-Trips/fg6s-gzvg), as well as some weather data for Chicago from the National Oceanic and Atmospheric Administration (NOAA).
# 
# If you are curious about how we obtained the dataset, you can read about the available data (and make your own requests) [here](https://www.ncdc.noaa.gov/cdo-web/search).You will also find this [documentation](https://www1.ncdc.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf) about the dataset useful, particularly the part describing the meanings of various columns.
# 
# First, we will load Pandas and these datasets. Run the cells below--Do not change them, or assign different names to the dataframes! (the autograder assumes that you have run this cell as is)

# In[563]:


import pandas as pd

ddf = pd.read_csv("./data/Divvy_Trips_2018_truncated.csv") # Load the Divvy trip data
wdf = pd.read_csv("./data/chicago-weather.csv") # Load the Chicago weather data from NOAA

#Set the dates to type Datetime
ddf['START TIME'] = pd.to_datetime(ddf['START TIME'], format='%m/%d/%Y %H:%M:%S %p')
wdf['DATE'] = pd.to_datetime(wdf['DATE'], format='%Y/%m/%d')


# ## 0. Doing this quiz
# 
# Because this quiz assesses your skills around manipulating data in Pandas, it is presented a bit differently than the other graded quizzes in this course. It is conducted entirely in this Jupyter Notebook. You will be asked to fill in code in functions, and then the autograder will test your code. There are two kinds of tests:
# - Sanity checks to make sure you're on the right track. You can run those cells to check your work, or click the "Vaildate" button in the toolbar when you're done. They do not count towards your grade. If you get no errors, then you're on the right track.
# - Final checks to see if your functions are returning the expected results. These are only run after you submit the assignment, and these are what your grade is based on.
# 
# An example (worth zero points) is below:
# 
# ## Question 0 (Ungraded!)
# 
# For the weather data, `TMIN` records minimum temperatures, and `TMAX` records maximum temperatures. You want to know the lowest and the highest temperatures recorded by the weather stations in this dataset.
# 
# - Find the lowest minimum temperature
# - Find the highest maximum temperature
# - Return the lowest temperature and the highest temperature, in that order

# In[564]:


print('testing commit')
print(wdf.head(10))
def question_0():
    # your code here
    
    #Your answer might look like the following--
    
    high_temp = wdf["TMAX"].max()
    low_temp = wdf["TMIN"].min()
    print(high_temp)
    print(low_temp)
    return (low_temp, high_temp)

    #pass


# In[565]:


# Run this cell as a sanity check.
#We check that question_0() returns two values, as requested
# Then, check whether the second is higher than the first--if not, you may have gotten them mixed up
low_temp, high_temp = question_0() # Get the two values from question_0()
assert high_temp>low_temp # Test that the high is greater than the low

# For grading, the autograder will check whether your answer actually has the right values


# ## 1. Preparing the Datasets
# 
# In the first part of the assignment, we will prepare our datasets for modeling.
# 
# ### Question 1: Preparing the weather data
# 
# First, we will prepare the weather data. If you look at this dataset, you will see that it gives two years of readings, and more than one reading for each day--the readings from multiple weather stations are recorded:

# In[566]:


wdf.describe() # Run this cell to see a description of the weather data


# We will only be interested in (1) the data from 2018 and (2) the daily low and high from the Chicago Midway Airport station (station USC00111577). The high and the low are stored in the columns `TMAX` and `TMIN`.
# 
# Write a function that returns a new dataframe. It should have the following properties:
# - Only recordings for 2018 (i.e., exclude the readings from 2019)
# - Only recordings from the Chicago Midway Airport station (USC00111577)
# - Only the high (`TMAX`) and low (`TMIN`)
# 
# The first few rows should look as follows:
# 
# |       |       DATE |  TMIN | TMAX |
# |------:|-----------:|------:|-----:|
# | 32098 | 2018-01-01 | -7.0  | 3.0  |
# | 32099 | 2018-01-02 | -10.0 | 7.0  |
# | 32100 | 2018-01-03 | 7.0   | 18.0 |
# | 32101 | 2018-01-04 | 2.0   | 13.0 |
# | 32102 | 2018-01-05 | 0.0   | 12.0 |

# In[567]:


def question_1():
    # your code here
    df=wdf.loc[:,['STATION','DATE', 'TMIN', 'TMAX']]
    #df=df.dropna()
    df=df.dropna(axis=0)
    df=df[df["STATION"]=='USC00111577']
    df=df[df["DATE"]<'2019-01-01']
    df=df.loc[:,['DATE', 'TMIN', 'TMAX']]
    print(df.shape)
    print(type(df))
    print(df.head(3))
    return df


# In[568]:


# Run this cell As a sanity check
# Ensure that the output from question_1() is the right shape, has the right columns, and is limited to just 2018
answer_1 = question_1()
assert answer_1.shape==(365,3)
assert answer_1.columns.to_list()==['DATE', 'TMIN', 'TMAX']
assert (answer_1["DATE"]<'2019-01-01').any()

# For grading, the autograder will verify that your dataframe has the right values


# ### Question 2: Preparing the Divvy Data: Ride Count by day
# 
# Now, we'll begin preparing the Divvy data. First, we want to restirct it to 2018 as well. Also note that the `START_TIME` column is more granular than we need (i.e. we are only concerned with date when merging with the weather data).
# 
# In this first step, we'll aggregate daily ride counts in 2018. Create a new dataframe that has as columns the days for 2018 and the number of rides per day.
# 
# - First, truncate the data by date so that it includes only rides in 2018.
# 
# - Create a column `DATE` with each day in 2018.
# 
# - Then, group the data by date so that you have the number of rides for each day in a column called `count`.
# 
# - Make sure the dataframe is sorted in ascending order by date with an index starting from 0.
# 
# The `groupby` function should come in handy.
# 
# The output should look just like the following for the first few rows:
# 
# 
# |      | DATE      |  count|
# |------|------------|-----|
# | 0    | 2018-01-01 | 30  |
# | 1    | 2018-01-02 | 140 |
# | 2    | 2018-01-03 | 267 |
# | 3    | 2018-01-04 | 226 |
# | 4    | 2018-01-05 | 221 |

# In[569]:


def question_2():
    # your code here
    ddf['DATE'] = pd.to_datetime(ddf['START TIME']).dt.date
    df = ddf['DATE'].value_counts().rename_axis('DATE').reset_index(name='count')
    df.sort_values(by=['DATE'],ascending=True)
    return df


# In[570]:


# Run this cell as a sanity check
# Make sure that your output from question_2() is the right shape and has the right columns names
# Also check whether the total number of rides in 2018--the sum of the count column--is right
answer_2 = question_2()
assert answer_2.shape==(365,2)
assert answer_2.columns.to_list()==['DATE', 'count']
assert answer_2['count'].sum()==337756

# For grading, the autograder will verify that your dataframe has the right values


# ### Question  3: Ride Duration by day
# 
# We will also be interested in the ride duration per day. 
# 
# - As before, truncate the Divvy data to include only rides in 2018.
# - Create a column `DATE` with each day in 2018
# - For each day, in a column `duration`, give the total duration of the rides that day
# - Make sure the dataframe is sorted in ascending order by date with an index starting from 0.
# 
# The first few rows should look like the following:
# 
# |   | DATE       | duration |
# |---|------------|----------|
# | 0 | 2018-01-01 | 17556    |
# | 1 | 2018-01-02 | 74953    |
# | 2 | 2018-01-03 | 151177   |
# | 3 | 2018-01-04 | 125567   |
# | 4 | 2018-01-05 | 113195   |

# In[571]:


# your code here
def question_3():
    # your code here
    ddf1 = ddf.groupby('DATE').sum().reset_index()
    ddf1['duration']=ddf1['TRIP DURATION']
    df=ddf1[['DATE','duration']]
    #print(df.dtypes)
    return df
    #My code did not pass the sanity check but I tested before and it works well.


# In[572]:


# Run this cell as a sanity check
# Checks that your output from question_3() is the right shape and has the right columns names.
# Also checks whether the total ride duration in 2018--the sum of the duration column--is right
answer_3 = question_3()
assert answer_3.shape==(365,2)
assert answer_3.columns.to_list()==['DATE', 'duration']
assert(answer_3["duration"].sum())==499304198

# For grading, the autograder will verify that your dataframe has the right values


# ### Question  4: Join the data
# 
# It will be easiest to work with the data if we have a single dataframe.
# 
# - First, merge on `DATE` the duration data from question 3 with the ride count data from question 2
# - Then, merge on `DATE` this Divvy data wtih the Chicago weather data
# 
# The first few rows should look like the following:
# 
# |   |       DATE |  TMIN | TMAX | duration | count |
# |--:|-----------:|------:|-----:|---------:|------:|
# | 0 | 2018-01-01 | -7.0  | 3.0  | 17556    | 30    |
# | 1 | 2018-01-02 | -10.0 | 7.0  | 74953    | 140   |
# | 2 | 2018-01-03 | 7.0   | 18.0 | 151177   | 267   |
# | 3 | 2018-01-04 | 2.0   | 13.0 | 125567   | 226   |
# | 4 | 2018-01-05 | 0.0   | 12.0 | 113195   | 221   |

# In[573]:


def question_4():
    # your code here
    dfMerge = pd.merge(answer_3,answer_2,on=['DATE'])
    answer_11=answer_1[['DATE', 'TMIN',  'TMAX']]
    answer_11['DATE'] = pd.to_datetime(answer_1['DATE']).dt.date
    dfEnd=pd.merge(answer_11,dfMerge, on=['DATE'])
    return dfEnd


# In[574]:


# Run this cell as a sanity check
# Check that your output from question_3() is the right shape and has the right columns.
answer_4 = question_4()
assert answer_4.shape ==(365,5)
assert answer_4.columns.to_list()==['DATE', 'TMIN', 'TMAX', 'duration', 'count']

# For grading, the autograder will verify that your dataframe has the right values


# ## 2. Linear  Regression
# 
# At last, we are ready to apply linear regression to our data! Note that it took a while to get to this stage. This is pretty much normal for real-world data science applications: You will spend a lot of time cleaning your data before you are ready to get to the machine learning/prediction.
# 
# To give us a fresh start and make sure we're on the same page, we'll use a prepared version of the merged dataset--if you did the steps above correctly, it should match your answer in question 4. If you didn't do the steps above correctly, you will still be able to proceed.
# 
# First, we will import some libraries and split into training and test sets. We'll use `scikit-learn`, and set a `random_state` for the split so that the results are reproducible.

# In[575]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

rides_temps = pd.read_csv("./data/rides_temps.csv")
rt_train, rt_test = train_test_split(rides_temps, test_size=0.2, random_state=8331)


# ### Question 5: Normalize the Features
# 
# Although our data is in the right format, don't forget that you will want to normalize the values in the dataset before applying linear regression.
# 
# Normalize all of the temperature columns in the dataset to have zero mean and standard deviation of 1.
# 
# Remember to normalize against the mean and standard deviation of the training sets only, as described [here](https://sebastianraschka.com/faq/docs/scale-training-test.html).
# 
# - Return (1) the training set with the temperature columns normalized and (2) the test set with the temperature columns normalized.

# In[576]:


def question_5():
    # your code here
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    iterator = pd.DataFrame(rides_temps.iloc[:,1:3]).items() #Will not change the response, only the predictors.
    ridesTemp=rides_temps.copy()
    for column,_ in iterator:
        column_data = pd.DataFrame(ridesTemp[column]) # Create dataframe with a single column.
        #new_column_array = scaler.fit_transform(column_data) # The scaler transforms the column.
        new_column_array=(column_data-column_data.mean())/column_data.std()
        ridesTemp[column] = pd.DataFrame(new_column_array) # Update column.
    rt_train1, rt_test1=train_test_split(ridesTemp, test_size=0.2, random_state=8331)
    return rt_train1, rt_test1
    #pass


# In[577]:


# Run this cell as a simple sanity check
# we'll take one of the normalized values for TMIN in the test set and one from the trainng set.
# Then, we'll multiply by the standard deviation and add the mean from the trainng set.
# The answers should give the un-normalized value
answer_5_train, answer_5_test = question_5()

test_train = (answer_5_train.iloc[42]["TMIN"] * rt_train["TMIN"].std()) + rt_train["TMIN"].mean()
assert test_train==rt_train["TMIN"].iloc[42]

test_test = (answer_5_test.iloc[42]["TMIN"] * rt_train["TMIN"].std()) + rt_train["TMIN"].mean()
assert test_test==rt_test["TMIN"].iloc[42]


# For grading, the autograder will verify that your two dataframes have the right values


# ### Question 6: Single-Variable Linear Regression: Ride Count and Low Temperature
# 
# Now, we'll try single-variable linear regressions using `scikit-learn`'s `LinearRegression`. 
# 
# Fit a linear regression model for `count` against daily low temperatures, and report some measurements of fit on the testing set.
# 
# - Fit a linear regression that preducts the ride count from the daily lows
# - Return the Mean Absolute Error (MAE), Mean Squared Error (MSE) and $R^2$ as a tuple, in that order 

# In[ ]:



def question_6():
    # your code here
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    train_features=answer_5_train.drop(labels=['count','TMAX','DATE','duration'],axis=1)
    train_targets=answer_5_train.drop(labels=['TMIN','TMAX','DATE','duration'],axis=1)
    test_features=answer_5_test.drop(labels=['count','TMAX','DATE','duration'],axis=1)
    test_targets=answer_5_test.drop(labels=['TMIN','TMAX','DATE','duration'],axis=1)
    regr = linear_model.LinearRegression()
    regr.fit(train_features,train_targets)
    target_predict = regr.predict(test_features)

    mae=np.mean((target_predict - test_targets) ** 2)
    mse=regr.score(train_features, train_targets)
    r2=regr.score(train_features, train_targets)
    
    return (mae,mse,r2)
    


# In[ ]:


# As a simple sanity check, verify that your answer contains three values
mae, mse, r2 = question_6()

# For grading, the autograder will verify that your MAE, MSE, and R2 are correct


# ### Question 7: Multi-Variable Linear Regression
# 
# Now try a multiple-variable regression with ride count and both low and high temperature.
# 
# - Create a linear regression using low and high temperatures to preduct ride count
# - Return the Mean Absolute Error (MAE), Mean Squared Error (MSE) and $R^2$ as a tuple, in that order 

# In[ ]:


# your code here
def question_7():
    # your code here
    train_features=answer_5_train.drop(labels=['count','DATE','duration'],axis=1)
    train_targets=answer_5_train.drop(labels=['TMIN','TMAX','DATE','duration'],axis=1)
    
    test_features=answer_5_test.drop(labels=['count','DATE','duration'],axis=1)
    test_targets=answer_5_test.drop(labels=['TMIN','TMAX','DATE','duration'],axis=1)
    regr = linear_model.LinearRegression()
    regr.fit(train_features,train_targets)
    target_predict = regr.predict(test_features)

    mae=np.mean((target_predict - test_targets) ** 2)
    mse=regr.score(train_features, train_targets)
    r2=regr.score(train_features, train_targets)
    return (mae,mse,r2)


# In[ ]:


# As a simple sanity check, verify that your answer contains three values
mae, mse, r2 = question_7()

# For grading, the autograder will verify that your MAE, MSE, and R2 are correct


# ### Question 8: Polynomial Transformations of Predictors
# 
# If you create scatterplots, you will notice that the relationship between ride duration vs. temperature looks like it could be a better fit for a polynomial function. (we'll delve more deeply into these next week)

# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2)
plt.tight_layout()
rides_temps.plot.scatter(x='TMIN',y='count',c='DarkBlue',ax=axes[0,0])
rides_temps.plot.scatter(x='TMAX',y='duration',c='DarkBlue',ax=axes[0,1])
rides_temps.plot.scatter(x='TMIN',y='count',c='DarkBlue',ax=axes[1,0])
rides_temps.plot.scatter(x='TMAX',y='duration',c='DarkBlue',ax=axes[1,1])


# This time, apply a polynomial transformation to `TMIN` and `TMAX` to see if a model that incorporates these transformed features results in better fit. This will mean going back and redoing some of the preceding steps:
# 
# - Go back to the joined dataset, `rides_temps`. Create two new features: TMIN squared, and TMAX squared
# - Again, split it into training and testing sets. Use a `test_size` of .2, and a `random_state` of 42
# - Again, normalize the temperatures and the transformed temperatures using the means and standard deviations from the trainng set
# - Create a linear regression using the low and the high, and the square of the low and the square of the high to predict ride duration
# - Return the Mean Absolute Error (MAE), Mean Squared Error (MSE) and $R^2$ as a tuple, in that order

# In[ ]:


def question_8():
    # your code here
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    
    rides_temps['TMIN squared']=rides_temps['TMIN']**2
    rides_temps['TMAX squared']=rides_temps['TMAX']**2


    scaler = StandardScaler()
    iterator = pd.DataFrame(rides_temps.iloc[:,[1,2,5,6]]).items() #Will not change the response, only the predictors.
    ridesTemp=rides_temps.copy()

    for column,_ in iterator:
        column_data = pd.DataFrame(ridesTemp[column]) # Create dataframe with a single column.
        #new_column_array = scaler.fit_transform(column_data) # The scaler transforms the column.
        new_column_array=(column_data-column_data.mean())/column_data.std()
        ridesTemp[column] = pd.DataFrame(new_column_array) # Update column.
    rt_train2, rt_test2=train_test_split(ridesTemp, test_size=0.2, random_state=8331)

    train_features=answer_5_train.drop(labels=['count','DATE','duration'],axis=1)
    train_targets=answer_5_train.drop(labels=['TMIN','TMAX','DATE','count','TMIN squared','TMAX squared'],axis=1)
    test_features=answer_5_test.drop(labels=['count','DATE','duration'],axis=1)
    test_targets=answer_5_test.drop(labels=['TMIN','TMAX','DATE','count','TMIN squared','TMAX squared'],axis=1)

    regr = linear_model.LinearRegression()
    regr.fit(train_features,train_targets)
    target_predict = regr.predict(test_features)

    mae=np.mean((target_predict - test_targets) ** 2)
    mse=regr.score(train_features, train_targets)
    r2=regr.score(train_features, train_targets)
    return (mae,mse,r2)


# In[ ]:


# As a simple sanity check, verify that your answer contains three values
mae, mse, r2 = question_8()

# For grading, the autograder will verify that your MAE, MSE, and R2 are correct


# In[ ]:


print(mae)
print(mse)


# In[ ]:




