# Recurrent Neural Network
# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
###################################################################
# Importing and cleaning the dataset
###################################################################

GDP = pd.read_csv('GDP.csv')
GDP = GDP.T
GDP=np.array(GDP)
  
#change '..' to nan
for r in range(GDP.shape[1]):
    itemindex = np.where((GDP[:,r]==('..')))
    GDP[itemindex,r]=np.nan
GDP_COLUMNS=GDP[0,:]  
GDP=GDP[5:,:].astype(np.float) #change strings to numbers

#change 0 to nan
zeroindex = []
for t in range(GDP.shape[1]):
    zeroindex = np.where((GDP[:,t] == (0)))
    GDP[zeroindex,t]=np.nan
       
#remove rows with more than 30 years of invalid data
to_keep=[]     
for u in range(GDP.shape[1]):    
    if (sum(np.isnan(GDP[:,u].astype(float)))<30):
        to_keep.append(u)      
GDP=GDP[:,to_keep]
GDP_COLUMNS=GDP_COLUMNS[to_keep]  
    
#remove coluns with unvariable data 
to_keep=[]     
for o in range(GDP.shape[1]):    
    if np.unique(GDP[:,o]).shape[0]>4 :
        to_keep.append(o)
GDP=GDP[:,to_keep]


GDP_COLUMNS=GDP_COLUMNS[to_keep] 
GDP_REAL_VALUES=GDP
GDP1=GDP

#inserting 5 np.nan in the end of each column to later making regression upon then
a=list(GDP)
b=[]
for i in range(GDP.shape[1]):
    b.append(np.nan)
 
for j in range(5):
    a.append(b)
    
a=np.array(a)
GDP=a      
GDP1=a               
np.argwhere(GDP_COLUMNS=='GDP growth (annual %)')
# difernt strategies to fill nan values
    
#4 5 7 9 11 13 15 17 100 exp
#6 sazonal decrescente    
    
#22 tem zeros de mais
#linear regression  

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import random

#making linear and exponential regression on missing data
#the next lines figure out the regression with minimal error and uses it
for i in range(GDP.shape[1]):
    y_train_lin=GDP[:,i]
    y_train_exp=GDP[:,i]
    x_train_lin=np.argwhere(~np.isnan(y_train_lin))
    if (x_train_lin.shape[0]<62): #check if there is data to be completed
        x_train_lin=np.argwhere(~np.isnan(y_train_lin))#index of not missing data
        x_test_lin=np.argwhere(np.isnan(y_train_lin)) #index missing data
        y_train_lin=y_train_lin[x_train_lin]#y of not missing data
        lm = LinearRegression() #making linear regression
        lm.fit(x_train_lin,y_train_lin)#fiting linear regression
        predictions_lin = lm.predict(x_test_lin) #using the model to fill missing data
        y_hat_lin = lm.predict(x_train_lin) #comparing the model to true values
        error_lin=y_hat_lin-y_train_lin #comparing the model to true values
        y_hat_dev_lin=[]
        for k in range(len(x_test_lin)):
            y_hat_dev_lin.append(random.uniform(-error_lin.std()*1.4,error_lin.std()*1.4)) #puting noise in data to be filled proportional to the standard deviation from true values
        y_validation_lin=[]    
        for e in range(len(x_train_lin)):
            y_validation_lin.append(random.uniform(-error_lin.std()*1.4,error_lin.std()*1.4)) #puting noise in validation data to be filled proportional to the standard deviation from true values   
        y_hat_dev_lin=np.array(y_hat_dev_lin) 
        y_validation_lin=np.array(y_validation_lin)
        predictions_lin=np.array(predictions_lin)
        y_validation_lin=np.reshape(y_validation_lin,(len(y_validation_lin), 1))
        y_hat_dev_lin=np.reshape(y_hat_dev_lin,(len(x_test_lin), 1))  
        y_validation_lin=abs(y_hat_lin+y_validation_lin)
        y_hat_noise_lin=abs(predictions_lin+y_hat_dev_lin)
        #plt.plot(x_test_lin,predictions_lin)
        #plt.plot(x_train_lin,y_hat_lin)
        #plt.scatter(x_train_lin,y_validation_lin)
        #plt.scatter(x_train_lin,y_train_lin)
        #plt.scatter(x_test_lin,y_hat_noise_lin)
    
        true_error_lin=(y_validation_lin-y_train_lin)
        squared_error_lin=true_error_lin**2
        lin_error=sum(squared_error_lin)
        #linear_regression_r2_score=r2_score(y_train,y_validation)
    
        #exponential regression
        x_train_exp=np.argwhere(~np.isnan(y_train_exp))
        if not(any(y_train_exp[x_train_exp]<0)):  #check if there is negative values 
            x_train_exp=np.argwhere(~np.isnan(y_train_exp))
            x_test_exp=np.argwhere(np.isnan(y_train_exp))
            y_train_exp=y_train_exp[x_train_exp]
            y_train_exp=np.log(y_train_exp)
            lm = LinearRegression()
            lm.fit(x_train_exp,y_train_exp)
            predictions_exp = lm.predict(x_test_exp)
            y_hat_exp = lm.predict(x_train_exp)
            error_exp=y_hat_exp-y_train_exp
            y_hat_dev_exp=[]
            for j in range(len(x_test_exp)):
                y_hat_dev_exp.append(random.uniform(-error_exp.std(),error_exp.std()))
            y_validation_exp=[]    
            for m in range(len(x_train_exp)):
                y_validation_exp.append(random.uniform(-error_exp.std(),error_exp.std()))    
            y_hat_dev_exp=np.array(y_hat_dev_exp) 
            y_validation_exp=np.array(y_validation_exp)
            predictions_exp=np.array(predictions_exp)
            y_validation_exp=np.reshape(y_validation_exp,(len(y_validation_exp), 1))
            y_hat_dev_exp=np.reshape(y_hat_dev_exp,(len(x_test_exp), 1))  
            y_validation_exp=y_hat_exp+y_validation_exp
            y_hat_noise_exp=predictions_exp+y_hat_dev_exp
            #plt.plot(x_test_exp,predictions_exp)
            #plt.plot(x_train_exp,y_hat_exp)
            #plt.scatter(x_train_exp,y_validation_exp)
            #plt.scatter(x_train_exp,y_train_exp)
            #plt.scatter(x_test_exp,y_hat_noise_exp)
            y_train_exp=GDP[:,i]
            #plt.scatter(x_train_exp,y_train_exp[x_train_exp])
            #plt.scatter(x_train_exp,np.exp(y_validation_exp))
            #plt.scatter(x_test_exp,np.exp(y_hat_noise_exp))
            #plt.plot(x_train_exp,np.exp(y_hat_exp))
            true_error_exp=(np.exp(y_hat_exp)-np.exp(y_validation_exp))
            squared_error_exp=true_error_exp**2
            exp_error=sum(squared_error_exp)
            #exp_regression_r2_score=r2_score(y_train[x_train_exp],y_validation_exp)
        if (any(y_train_exp[x_train_exp]<0)): 
            GDP1[x_test_lin,i]=y_hat_noise_lin
            
        if exp_error<=lin_error: #check wich regression has the smallest error
            #GDP[x_test_exp,i]=np.exp(y_haceedt_noise_exp)
            GDP1[x_test_exp,i]=np.exp(y_hat_noise_exp)
        if exp_error>lin_error:
            #GDP[x_test_lin,i]=np.exp(y_hat_noise_lin)
            GDP1[x_test_lin,i]=y_hat_noise_lin
            #plt.scatter(range(57),GDP[:,i])
            #plt.scatter(range(57),GDP1[:,i])
            
#float(lm.coef_)
#float(lm.intercept_)
#isotonic regression

#import seaborn as sns
#sns.distplot(USAhousing['Price'])
#sns.pairplot(USAhousing)
#np.savetxt("foo.csv", a, delimiter=",")
#child=pd.DataFrame(data=GDP[4:,20],index=GDP[4:,20], columns=GDP[1,20])
    
    
#BRA=np.argwhere(GDP == 'BRA') 
#GDP_flatten=[]  
#for i in range(0,268):  
#    GDP_flatten=np.concatenate((GDP_flatten,GDP[4:,i]), axis=0)   
Nvar=GDP1.shape[1] #Number of variables used as input in the model
windowsize=12; #Time window in wich the model will look 
GDP_BRA=GDP1

# Feature Scaling
Mean = []
STD = []
MAX = [] 
MIN = []
for i in range(GDP.shape[1]):
    Mean.append(GDP_BRA[:,i].mean())
    STD.append(GDP_BRA[:,i].std())
    GDP_BRA[:,i]=(GDP_BRA[:,i]-Mean[i])/STD[i]
    MAX.append(max(GDP_BRA[:,i]))
    MIN.append(abs(min(GDP_BRA[:,i])))
    GDP_BRA[:,i]=(GDP_BRA[:,i]+MIN[i])/(MAX[i]+MIN[i]) 
    
    
# Creating a data structure with 32 timesteps (x) for each 1 output (y)
data_set=GDP_BRA
X_set = []
y_set = []
GDP_index=int(np.argwhere(GDP_COLUMNS=='GDP growth (annual %)'))       
for years in range(windowsize, data_set.shape[0]-5):
    X_set.append(data_set[years-windowsize:years,:])       
for years in range(windowsize, data_set.shape[0]-5):
    y_set.append(data_set[years,GDP_index])
        
X_set, y_set = np.array(X_set), np.array(y_set)
# Reshaping
y_train=y_set

X_train = np.reshape(X_set, (X_set.shape[0],windowsize,Nvar))

###################################################################
# Part 2 - Building the RNN
###################################################################

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
batch = 1
# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = int(X_train.shape[2]/windowsize) , return_sequences = True, input_shape = (X_train.shape[1], Nvar)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = int(X_train.shape[2]/windowsize), return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = int(X_train.shape[2]/windowsize)))
regressor.add(Dropout(0.2))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.load_weights('BRA-l2-nl444-w32-e100-bmin-d01-Perfeito.h5')

# Fitting the RNN to the Training set
#regressor.fit(X_train, y_train, epochs = 1000, batch_size = int(batch)) 

###################################################################
# Part 3 - Making the predictions and visualising the results
###################################################################

# Getting the real GDP from last years

data_set=GDP_BRA

X_test = []
Y_test = []

GDP_index=int(np.argwhere(GDP_COLUMNS=='GDP growth (annual %)'))       
for years in range(data_set.shape[0]-5,data_set.shape[0]):
        X_test.append(data_set[years-windowsize:years,:])
        
for years in range(data_set.shape[0]-5, data_set.shape[0]):
        Y_test.append(data_set[years,GDP_index])
X_test, Y_test  = np.array(X_test), np.array(Y_test)
# Reshaping
y_train=y_set

X_test = np.reshape(X_test, (5,windowsize,Nvar))       

predicted_GDP = regressor.predict(X_train)
predicted_GDP=(predicted_GDP*(MAX[GDP_index]+MIN[GDP_index])-MIN[GDP_index]) 
predicted_GDP=predicted_GDP*STD[GDP_index]+Mean[GDP_index]

####################################
real_GDP=GDP_REAL_VALUES[GDP_REAL_VALUES.shape[0]-len(predicted_GDP):,GDP_index]
# Visualising the results
x_year=[]
X_init_year=1994
X_final_Year=2018
for i in range(len(predicted_GDP)):
    x_year.append(X_final_Year+1-len(predicted_GDP)+i)   

plt.ylim(int(min(min(real_GDP),min(predicted_GDP)))-1,int(max(max(real_GDP),max(predicted_GDP)))-3)   
plt.xlim(X_init_year,X_final_Year)  
plt.yticks(range(-5,10,1), [str(x) + "%" for x in range(-5, 10, 1)], fontsize=14)    
plt.xticks(fontsize=12,rotation=45) 
for y in range(-4,10,1):    
    plt.plot(range(X_init_year,X_final_Year+1), [y] * len(range(X_init_year,X_final_Year+1)), "--", lw=0.5, color="black", alpha=0.3) 
plt.plot(x_year,real_GDP, color = 'red', label = 'Real Brazil GDP')
plt.plot(x_year,predicted_GDP, color = 'blue', label = 'Predicted Brazil GDP')
plt.tick_params(axis="both", which="both", bottom="off", top="off",labelbottom="on", left="off", right="off", labelleft="on") 
plt.title('Brazil GDP Prediction')
plt.xlabel('Year')
plt.ylabel('Brazil GDP growth')
plt.legend(loc='lower center', frameon=True)
plt.savefig('CaboVerde-18.png',bbox_inches='tight', dpi=800)

#Using the model to predict the next years (dynamic forecasting)

data_set=GDP_BRA
X_test = []
Y_test = []
GDP_index=int(np.argwhere(GDP_COLUMNS=='GDP growth (annual %)'))       
for years in range(data_set.shape[0]-5,data_set.shape[0]):
    X_test.append(data_set[years-windowsize:years,:])       
for years in range(data_set.shape[0]-5, data_set.shape[0]):
    Y_test.append(data_set[years,GDP_index])
X_test, Y_test  = np.array(X_test), np.array(Y_test)
# Reshaping
y_train=y_set
X_test = np.reshape(X_test, (5,windowsize,Nvar))       
predicted_GDP = regressor.predict(X_test)
predicted_GDP=(predicted_GDP*(MAX[GDP_index]+MIN[GDP_index])-MIN[GDP_index]) 
predicted_GDP=predicted_GDP*STD[GDP_index]+Mean[GDP_index]

x_year=[]
for i in range(len(predicted_GDP)):
    x_year.append(2019+i)   
   
plt.xlim(2019, 2023)  
plt.yticks(range(-int(min(predicted_GDP))-1,int(max(predicted_GDP))+1,1), [str(x) + "%" for x in range(-int(min(predicted_GDP)), int(max(predicted_GDP))+1, 1)], fontsize=14)    
plt.xticks(fontsize=12,rotation=45) 
for y in range(-4,10,1):    
    plt.plot(range(2019, 2018 + len(predicted_GDP)), [y] * len(range(2019, 2018 + len(predicted_GDP))), "--", lw=0.5, color="black", alpha=0.3) 
plt.plot(x_year,predicted_GDP, color = 'blue', label = 'Predicted Brazil GDP')
plt.tick_params(axis="both", which="both", bottom="off", top="off",labelbottom="on", left="off", right="off", labelleft="on") 
plt.title('Brazil GDP Prediction')
plt.xlabel('Year')
plt.ylabel('Brazil GDP growth')
plt.legend(loc='lower center', frameon=True)



X_val=[]
for years in range(data_set.shape[0]-windowsize, data_set.shape[0]):
        X_val.append(data_set[years-windowsize:years,:])
yhat =[]
y=0
X_use=[] 

for i in range(1,9,1):
    X_imp = np.reshape(X_use, (1,32,1))
    y=float(regressor.predict(X_imp))
    yhat.append(y)
    X_use.append(y)
    X_use = X_use[1:]
    
yhat=np.array(yhat)
yhat=(yhat*(MAX[GDP_index]+MIN[GDP_index])-MIN[GDP_index]) 
yhat=yhat*STD[GDP_index]+Mean[GDP_index]

#visualizing the result
x_year=[]
for i in range(len(predicted_GDP)):
    x_year.append(2027-len(predicted_GDP)+i)   
plt.plot(x_year,predicted_GDP, color = 'red', label = 'Expected Brazil GDP')
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
plt.savefig('BRA_FUTo.png',bbox_inches='tight', dpi=850)

