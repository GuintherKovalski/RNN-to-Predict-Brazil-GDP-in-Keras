This RNN model try to predict the Brazil GDP for the next 8 years!

Using the dataset from the World Bank with data from over 269 nations from the last 56 years.

More than predict absolut values, this kind of neural network is usefull to predict tendences.

Here is the result for the single var model (LSTM1):



And here is the result for the multivariable model (LSTM2) with 100 epochs:
![alt text](https://raw.githubusercontent.com/GuintherKovalski/RNN-to-Predict-Brazil-GDP-in-Keras/master/BRAZIL.png)

and with 1000

![alt text](https://raw.githubusercontent.com/GuintherKovalski/RNN-to-Predict-Brazil-GDP-in-Keras/master/BRA82-18.png)


to use the multivariable model a lot of preprocessing was made, one of then is complete the missing data. To do it i used logarithm, exponential and linear regression, and used the one with the smaller error. Some of the completed data is the follows:

I also predicted the GDP for the next years. This is the result with the single var model:






