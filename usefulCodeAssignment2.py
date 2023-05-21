print('BEFORE TRUNCATE')
print(wdf.shape)
print(wdf.head(5))
print('---------------------------------')
print('AFTER TRUNCATE')
df=wdf.loc[:,['STATION','DATE', 'TMIN', 'TMAX']]
print(df.shape)
print('---------------------------------')
print('AFTER CLEANING')
df2=df.dropna()
df2=df.dropna(axis=0)
df3=df2[df2["STATION"]=='USC00111577']
df4=df3[df3["DATE"]<'2019-01-01']
print(df4.shape)
print(df4.head(4))



print('testing')
# print(ddf.tail(2))
# ddf['Dates'] = pd.to_datetime(ddf["START TIME"]).dt.date
# ddf['Time'] = pd.to_datetime(ddf["START TIME"]).dt.time
ddf['START TIME'] = pd.to_datetime(ddf['START TIME'], format='%m/%d/%Y)
print(ddf.shape)
# ddf=ddf[ddf["START TIME"]<'2019-01-01']
print(ddf.shape)
print(ddf.describe())
def question_2():
    # your code here
    pass
    
    
    
    
    
    
    
    answer question 2
    ddf['DATE'] = pd.to_datetime(ddf['START TIME']).dt.date
df = ddf['DATE'].value_counts().rename_axis('DATE').reset_index(name='count')
df.sort_values(by=['DATE'],ascending=True)
print (df.head(6))
print(df.shape)
    
    
    
    
    
    
    # ddf['DATE'] = pd.to_datetime(ddf['START TIME'], format='%m/%d/%Y')
ddf['DATE'] = pd.to_datetime(ddf['START TIME']).dt.date
# ddf['count']=ddf['DATE'].value_counts()
# print(ddf['DATE'].value_counts())
# print(ddf['DATE'].unique())

# df = pd.DataFrame(ddf['DATE'].unique(), columns=['DATE'])
# df.sort_values(by=['DATE'],ascending=False)

df=ddf['DATE'].value_counts()

df = ddf['DATE'].value_counts().rename_axis('DATE').reset_index(name='count')
df.sort_values(by=['DATE'],ascending=True)
print (df.head(6))


# print(ddf['DATE'].value_counts())
# print(ddf['DATE'].value_counts())

# df = pd.DataFrame(ddf['DATE'].value_counts(), columns=['DATE','count'])


# df.sort_values(by=['DATE'],ascending=False)
# print(df.head(10))


# print(df.head(10))
# ddf = ddf.groupby(['DATE']).count()
# ddf.sort_values(by=['DATE'],ascending=False)
# print(ddf.tail(10))



# your code here
# print(ddf.head(1))
# print(ddf.dtypes)
ddf['DATE'] = pd.to_datetime(ddf['START TIME']).dt.date
# df = ddf['TRIP DURATION'].value_counts().rename_axis('DATE').reset_index(name='count')

# ddf.set_index('DATE', inplace=True)

# ddf = ddf.groupby(['DATE','TRIP DURATION']).count()

ddf = ddf.groupby('DATE').sum().reset_index()
ddf['duration']=ddf['TRIP DURATION']
# Using reset_index()

print(ddf.head(10))




ddf['DATE'] = pd.to_datetime(ddf['START TIME']).dt.date
ddf = ddf.groupby('DATE').sum().reset_index()
ddf['duration']=ddf['TRIP DURATION']
df=ddf[['DATE','duration']]
print(df.head(10))
print(df.shape)
print(df.columns.to_list())
assert df.columns.to_list()==['DATE', 'duration']
assert(df["duration"].sum())==499304198
print('fine')




question 4
# # print(answer_11.keys)
# print(answer_11.head(2))
# print(answer_2.head(2))
# print(answer_3.head(2))

# print(answer_2.shape)
# print(answer_3.shape)
# print(answer_11.shape)

dfMerge = pd.merge(answer_3,answer_2,on=['DATE'])
# print('///////////////')
# print(dfMerge.shape)
# print(answer_11.shape)
# print(type(dfMerge))
# print(type(answer_11))

# print(dfMerge.head(3))
# print(answer_11.head(3))
answer_11=answer_1[['DATE', 'TMIN',  'TMAX']]
answer_11['DATE'] = pd.to_datetime(answer_1['DATE']).dt.date

# print(answer_3.dtypes)
# print(answer_2.dtypes)
# print(answer_11.dtypes)
# print(dfMerge.dtypes)



dfEnd=pd.merge(answer_11,dfMerge, on=['DATE'])
print(dfEnd.head(2))






question 5

# print(rides_temps.shape)
# print(rides_temps.dtypes)
# # print(rides_temps.keys)
print(rides_temps.head(6))
print(rides_temps.describe().loc[['mean', 'std']])


print(' ')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
print(' ')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
iterator = pd.DataFrame(rides_temps.iloc[:,1:3]).items() #Will not change the response, only the predictors.
ridesTemp=rides_temps.copy()
# your code here
for column,_ in iterator:
    column_data = pd.DataFrame(rides_temps[column]) # Create dataframe with a single column.
    new_column_array = scaler.fit_transform(column_data) # The scaler transforms the column.
    ridesTemp[column] = pd.DataFrame(new_column_array) # Update column.
rt_train1, rt_test1=train_test_split(ridesTemp, test_size=0.2, random_state=8331)

# print(rt_train1.head(3))
# print(rt_train1["TMIN"].std())
# print(rt_train1["TMIN"].mean())

# print(rt_train1.iloc[42]["TMIN"])
# print((rt_train1.iloc[42]["TMIN"] * rt_train1["TMIN"].std()) + rt_train1["TMIN"].mean())


test_train = (rt_train1.iloc[42]["TMIN"] * rt_train1["TMIN"].std()) + rt_train1["TMIN"].mean()
print(test_train)
print(rt_train1["TMIN"].iloc[42])
assert test_train==rt_train1["TMIN"].iloc[42]







# (X - X.mean())/X.std()


# your code here
for column,_ in iterator:
    column_data = pd.DataFrame(rides_temps[column]) # Create dataframe with a single column.
    #new_column_array = scaler.fit_transform(column_data) # The scaler transforms the column.

    new_column_array=(column_data-column_data.mean())/column_data.std()
    ridesTemp[column] = pd.DataFrame(new_column_array) # Update column.
rt_train1, rt_test1=train_test_split(ridesTemp, test_size=0.2, random_state=8331)


test_train = (rt_train1.iloc[42]["TMIN"] * rt_train1["TMIN"].std()) + rt_train1["TMIN"].mean()

print(test_train)

tmp=rt_train1["TMIN"].iloc[42]
print(tmp)









train_features,train_targets=answer_5_train['count'],answer_5_train['TMIN']

# print(type(train_features))
print(train_features.head(3))
print(train_targets.head(3))


print('-----')

print('-----')
train_features = answer_5_train.loc[:,'count']
train_targets = answer_5_train.loc[:,'TMIN']

# print(type(train_features))
print(train_features.head(3))
print(train_targets.head(3))

from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
regr.fit(train_features,train_targets)












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



# The mean squared error and RSS (by hand)
print("Mean squared error: %.2f" % np.mean((target_predict - test_targets) ** 2))
# The residual sum of squares (RSS)
print("RSS: %.2f" % np.sum((target_predict - test_targets) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(train_features, train_targets))

mae=np.mean((target_predict - test_targets) ** 2)
mse=regr.score(train_features, train_targets)
r2=regr.score(train_features, train_targets)

