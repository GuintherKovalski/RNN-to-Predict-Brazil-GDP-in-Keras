This RNN model try to predict the Brazil GDP for the next 8 years!

Using the dataset from the World Bank with data from over 269 nations from the last 56 years, with 1600 variables.

More than predict absolut values, this kind of neural network is usefull to predict tendences.

Here is the result for the single variable model (LSTM1):

![alt text](https://raw.githubusercontent.com/GuintherKovalski/RNN-to-Predict-Brazil-GDP-in-Keras/master/BRA1.png)

And here is the result for the multivariable model (LSTM2) with 100 epochs:
![alt text](https://raw.githubusercontent.com/GuintherKovalski/RNN-to-Predict-Brazil-GDP-in-Keras/master/BRAZIL.png)

and with 1000 epochs:

![alt text](https://raw.githubusercontent.com/GuintherKovalski/RNN-to-Predict-Brazil-GDP-in-Keras/master/BRA82-18.png)


to use the multivariable model a lot of pre processing was made, one of then is complete the missing data. To do it i used logarithm, exponential and linear regression, and chose the one with the smaller error. Some of the completed data are the follows:

![alt text](https://raw.githubusercontent.com/GuintherKovalski/RNN-to-Predict-Brazil-GDP-in-Keras/master/Agriculture%2C%20forestry%2C%20and%20fishing%2C%20value%20added%20(constant%202010%20US%24).png)

![alt text](https://raw.githubusercontent.com/GuintherKovalski/RNN-to-Predict-Brazil-GDP-in-Keras/master/Cereal%20production%20(metric%20tons).png)

I also predicted the GDP for the next years. This is the result with the single var model:

![alt text](https://raw.githubusercontent.com/GuintherKovalski/RNN-to-Predict-Brazil-GDP-in-Keras/master/BRA22_FUT1.png)

The official prediction of GDP from Brazilian central bank is 2.5%

To do:

  Split data in train/test/validation

  Create a lib for data completion instead of using it inside the main algorithm.
  
  Formating the data and make the prediction for the next years with the multi var model.
  
  Create and train a model with fewer variables, with more parcimony. Need some help to chose. 

