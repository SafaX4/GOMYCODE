#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df=pd.read_csv("C:/Users/EXTRA/Downloads/iris.csv")
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

x=df[["sepal.length","sepal.width","petal.length","petal.width"]]
y=df["variety"]
y2=np.array(y).reshape(-1,1)
encoder = OneHotEncoder(sparse=False)
y1=encoder.fit_transform(y2)
x_train,x_test,y_train,y_test = train_test_split(x,y1,test_size=0.20,random_state=40)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
model = Sequential()
model.add(Dense(10,input_dim=4,activation='relu'))
model.add(Dense(3,input_dim=10,activation='softmax'))
op=Adam(lr=0.005)
model.compile(loss="categorical_crossentropy",optimizer=op,metrics=["accuracy"])
model.fit(x_train,y_train,epochs=1000,verbose=2,batch_size=20)

results = model.evaluate(x_test,y_test)
print(results)

