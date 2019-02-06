# Recurrent Neural Network
# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing and cleaning the dataset
GDP = pd.read_csv('GDP_only.csv')
GDP.fillna(0,inplace=True)
GDP = GDP.T
GDP=np.array(GDP)
for i in range(4,GDP.shape[0]):
    itemindex = np.where(GDP[i,:]==('..'))
    GDP[i,itemindex]=0
    GDP1=GDP[i,:]
    GDP1=GDP1.astype(np.float)
    nonzeroindex = GDP1.nonzero()
    Mean=np.mean(GDP1[nonzeroindex])
    zeroindex=np.argwhere(GDP1 == 0)
    GDP1[zeroindex]=Mean
    GDP[i,:]=GDP1   
BRA=np.argwhere(GDP == 'BRA') 
GDP_flatten=[]  
for i in range(0,268):  
    GDP_flatten=np.concatenate((GDP_flatten,GDP[4:,i]), axis=0)   
Nvar=1
windowsize=32;
GDP_BRA=GDP[6:,26]
# Feature Scaling
Mean=GDP_BRA.mean()
STD=GDP_BRA.std()
MIN=abs(min([n for n in GDP_BRA if n<0]))
MAX=max(GDP_BRA)+MIN
GDP_BRA=(GDP_BRA-Mean)/STD
GDP_BRA1=(GDP_BRA+MIN)/MAX
# Creating a data structure with 8 timesteps and 1 output
training_set=GDP_BRA1
X_train = []
y_train = []
for i in range(windowsize, training_set.shape[0]):
    X_train.append(training_set[i-windowsize:i])
    y_train.append(training_set[i])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Part 2 - Building the RNN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
batch = int((GDP[6:,26].shape[0]-windowsize))
# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = int(windowsize*4) , return_sequences = True, input_shape = (X_train.shape[1], Nvar)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = int(windowsize*4), return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = int(windowsize*4)))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 5000, batch_size = int(batch)) #100 32
# Part 3 - Making the predictions and visualising the results
# Getting the real GDP from last 8 years
GDP_BR=GDP[6:,26]#GDP[5:,26]=BRA
GDP_BRA=(GDP_BRA-Mean)/STD
GDP_BRA=(GDP_BR+MIN)/MAX
GDP_BRA=GDP_BRA
inputs = GDP_BRA
years=GDP[6:,26].shape[0]-windowsize  #anos a serem previstos
inputs1 = inputs[inputs.shape[0]-years-windowsize:]
X_test = []
for i in range(windowsize,inputs1.shape[0]):
    X_test.append(inputs[i-windowsize:i])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], Nvar))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price= predicted_stock_price
predicted_stock_price=predicted_stock_price*MAX-MIN
predicted_stock_price=predicted_stock_price*STD+Mean
####################################
real_stock_price=GDP_BR[GDP_BR.shape[0]-predicted_stock_price.shape[0]:]
# Visualising the results
x_year=[]
for i in range(len(predicted_stock_price)):
    x_year.append(2019-len(predicted_stock_price)+i)   
plt.plot(x_year,real_stock_price, color = 'red', label = 'Real Brazil GDP')
plt.plot(x_year,predicted_stock_price, color = 'blue', label = 'Predicted Brazil GDP')
plt.ylim(0,20)    
plt.xlim(2005, 2018)  
plt.yticks(range(-5,9,1), [str(x) + "%" for x in range(-5, 9, 1)], fontsize=14)    
plt.xticks(fontsize=14,rotation=45) 
for y in range(-4,9,1):    
    plt.plot(range(2005, 2019), [y] * len(range(2005, 2019)), "--", lw=0.5, color="black", alpha=0.3)   
plt.tick_params(axis="both", which="both", bottom="off", top="off",labelbottom="on", left="off", right="off", labelleft="on") 
plt.title('Brazil GDP Prediction')
plt.xlabel('Year')
plt.ylabel('Brazil GDP growth')
plt.legend(loc='botton', frameon=True)
plt.savefig('chi.png',bbox_inches='tight', dpi=800)

#Using the model to predict the next 4 years (dynamic forecasting)
yhat =[]
y=0
X_use=[] 
for i in range(0,inputs[24:56].shape[0],1): 
  X_use.append(inputs[(24+i)])
  

for i in range(1,9,1):
    X_imp = np.reshape(X_use, (1,32,1))
    y=float(regressor.predict(X_imp))
    yhat.append(y)
    X_use.append(y)
    X_use = X_use[1:]
yhat=np.array(yhat)
yhat=yhat*MAX-MIN
yhat=yhat*STD+Mean

#visualizing the result
x_year=[]
for i in range(len(yhat)):
    x_year.append(2027-len(yhat)+i)   
plt.plot(x_year,yhat, color = 'red', label = 'Prevision Brazil GDP')
plt.ylim(0,5)    
plt.xlim(2019, 2026)  
plt.yticks(range(0,5,1), [str(x) + "%" for x in range(0, 5, 1)], fontsize=14)    
plt.xticks(fontsize=14,rotation=45) 
for y in range(0,5,1):    
    plt.plot(range(2019, 2027) , [y] * len(range(2019, 2027) ), "--", lw=0.5, color="black", alpha=0.3)   
plt.tick_params(axis="both", which="both", bottom="off", top="off",labelbottom="on", left="off", right="off", labelleft="on") 
plt.title('Brazil GDP Prediction')
plt.xlabel('Year')
plt.ylabel('Brazil GDP growth')
plt.legend(loc='botton', frameon=True)
plt.savefig('chi_FUT1.png',bbox_inches='tight', dpi=800)

regressor.save_weights('IN-l2-nl444-w32-e10000-bTOTAL-d01-minmaxscaler.h5')
