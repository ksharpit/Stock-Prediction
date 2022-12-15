#!/usr/bin/env python
# coding: utf-8

# ### Keras and Tensorflow 2.0

# In[4]:


# data collection

# 1. Collect the stock data -- AAPL (APPLE)
# 2. Preprocess the data - Train and Test
# 3. Create the stacked LSTM model (Long-short term memory)
# 4. Then we will train the model 
# 5. Predict the test data and plot the output
# 6. Finally we will predict the future 30 days and plot the output


# In[5]:


import os
import pandas_datareader as pdr


# In[6]:


key = 'c806b4a310721626a2b28ee28ad638381bf1c9b5'
df = pdr.get_data_tiingo('AAPL', api_key=key)
df.head()
# key name is diff for everyone 
# get_data_tiingo to get the data


# In[7]:


df.to_csv("AAPL.csv")


# In[8]:


df.head()


# In[9]:


import pandas as pd


# In[10]:


df=pd.read_csv("AAPL.csv")


# In[11]:


df.head()
#reading in csv


# In[12]:


df.tail()
# from the end


# In[13]:


df1=df.reset_index()['close']
#doing stock prediction for 'close' column


# In[14]:


df1.shape
# shows no. of records under close column
# 1258 rows and 1 close column 


# In[15]:


df1


# In[16]:


import matplotlib.pyplot as plt
plt.plot(df1)


# ### LSTM is very sensitive to the scale of data therefore we use minmax scalar

# In[17]:


import numpy as np


# In[18]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1)) #converting in between 0 to 1
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[19]:


df1


# In[20]:


print(df1) #prints list i.e. a dataset


# ### Splitting dataset into train and test split

# In[21]:


# ways to do train test split
# crossvalidation
# random seed
# BUt these 2 are valid only for linear and logistic based problem statement
# In case of "time series data", we divide data set in a diff way


# In[22]:


# We divide data as next data depends on previous day data
# example: the data of day 2 depends on day 1. 
# 65% of the data for training
# 35% for test
# After doing data division, we do the data preprocessing


# In[23]:


training_size = int(len(df1)*0.65) # Train data
test_size = len(df1)-training_size  # Test data
train_data, test_data = df1[0:training_size,:],df1[training_size:len(df1):1]


# In[24]:


training_size,test_size


# In[25]:


train_data, test_data


# In[26]:


# Data Preprocessing
# To compute output of next day, how many previous days we have to consider
# if timesteps = 3 i.e. no of previoud days data
# Time series data : 120,130,125,140,134,150,160,170
# Train : 120,130,125,140,134,150,160, 190, 154
# Test: 160, 190, 154, 160, 170
# Timesteps = 3 i.e. 3 features
# Indep features  Depen features
# X_train         Y_train              y-train f1 f2 f3     o/p(y-test)
# f1  f2  f3      O/P                         160 190 154   160
# 120 130 125     140                         190 154 160   170
# 130 125 140     134


# In[27]:


# Now training

import numpy as np
# Convert an array of values into a dataset matrix

def create_dataset(dataset, time_step = 1): #timestep is 1 by default 
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1): # leaving last 100 dates
         dataX.append(dataset[i:(i+time_step),0]) # at i=0 iteration, for timestep =3, 0,1,2,3 positions data are appended to dataX, as 3rd position is last as seen in range last no. not included
         dataY.append(dataset[i+time_step,0]) # 4th position data is appended to dataY
    return np.array(dataX), np.array(dataY)


# In[28]:


time_step = 100
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)


# In[29]:


print(X_train)


# In[30]:


print(Y_train)


# In[31]:


print(X_train.shape), print(Y_train.shape)
# 100 are features


# In[32]:


print(X_test.shape), print(Y_test.shape)


# In[33]:


# Now creating stack LSTM Model
# Before going into LSTM model, we need to reshape X-train and X-test into 3-D array
# Now X_train and X_test are 2-D right now, therefore we add 1 to make it 3-D


# In[34]:


# Reshape input to be [samples,time steps,features] which is required for LSTM

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
# (X_train.shape[1], 1) will be given as input to the LSTM
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], 1)


# In[35]:


# Create stacked LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[36]:


model=Sequential()
# stacked LSTM 1 after another
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
# 100 features
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer = 'adam')


# In[37]:


model.summary()


# In[38]:


model.fit(X_train,Y_train,validation_data=(X_test, Y_test),epochs=100,batch_size=64,verbose=1)


# In[39]:


# lets do the prediction and check performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[40]:


# Transform back to original form as RSME (root mean square error) requires original data
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[41]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(Y_train,train_predict))


# In[42]:


### Test Data RMSE
math.sqrt(mean_squared_error(Y_test,test_predict))


# In[43]:


### Plotting 
# shift train predictions for plotting
look_back=100 # time step = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)                                                                                                 
plt.show()

# Green is the predicted output for the test data
# Blue is complete dataset
# Yellow is how the prediction has gone for the training data


# In[45]:


# predicting next 30 days data
len(test_data)
#len(test_data)-look_back


# In[47]:


# for 23 date, 100 days before 22 date's data is taken and reshaped
# len(test_data)-1=440-100=340
x_input=test_data[len(test_data)-look_back:].reshape(1,-1)
x_input.shape


# In[49]:


#converting into list
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input
#printing all previous 100 days


# In[50]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)
#prints all the output for next 30 days


# In[51]:


import numpy as np
day_new=np.arange(1,101) #new data that starts from 1 to 101

day_pred=np.arange(101,131) #new pred data from 101 to 131 = 30 days 


# In[52]:


import matplotlib.pyplot as plt


# In[53]:


len(df1)


# In[54]:


plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
#from 1157 to 1257 it will display the realtime data  for 100 days
plt.plot(day_pred,scaler.inverse_transform(lst_output))

#ornage colour is the predicted 30 days output


# In[55]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[68]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as data
from keras.models import load_model
#import streamlit as st

# start = '2010-01-01'
# end= '2022-10-10'

# st.title("Stock market trend")

# user_input = st.text_input("Enter Stock Ticker",'AAPL')
# df = data.DataReader(user_input,'yahoo',start,end)


# st.subheader("data from 2010 to 2022")
# st.write(df.describe())

import pandas_datareader.data as web
import datetime    

start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2016, 1, 27)
df = web.DataReader("GOOGL", 'yahoo', start, end)

dates =[]
for x in range(len(df)):
    newdate = str(df.index[x])
#     print(newdate)
    newdate = newdate[0:10]
#     print(newdate)
    dates.append(newdate)
#   print(dates)

df['dates'] = dates

print(df.head())
print(df.tail())
print(len(df))


# In[60]:


df


# In[ ]:




