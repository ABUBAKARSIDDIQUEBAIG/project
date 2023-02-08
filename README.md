# project
My Project is here  
Tesla   Stock price    Prediction
Subject
Data Mining 


ABUBAKARSIDDIQUE Baig
	INSTRUCTIONS

Implementation of Stock Price Prediction in Python
1. Importing Modules
2. Loading and Preparation of Data
3. Understanding the Data
3.1 Getting Unique Stock Names
3.2 Extracting Data for a specific stock name
3.3 Visualizing the stock data
4. Creating a new Data frame and Training data
5. Building LSTM Model
6. Compiling the Model
7. Testing the model on testing data
8. Error Calculation
9. Make Predictions
10. The Actual vs Predicted Values

For the project, we will be using basic modules like 
NumPy,
 pandas, and
 matplotlib. 
In addition to this, we will be using some submodules of karas to create and build our model properly.
We would also require the math module for basic calculation and preprocessing module of sk learn to handle the data in a better and simpler way.
1. Importing Modules
First step is to import all the necessary modules in the project.
2. Loading and Preparation of Data3. Understanding the Data
3.1 Getting Unique Stock Names
3.2 Extracting Data for a specific stock name
We will try to understand how the stock data works by taking an input of a stock name from the user and collecting all data of that particular stock name.

To visualize the data we will be first plotting the date vs close market prices for the FITB stock for all the data points.

3.3 Visualizing the stock data
To visualize the data we will be first plotting the date vs close market prices for the FITB stock for all the data points.
4. Creating a new Dataframe and Training data
To make our study easier we will only consider the closing market price and predict the closing market price using Python. The whole train data preparation is shown in the steps below. Comments are added for your reference.
5. Building LSTM Model
The LSTM model will have two LSTM layers with 50 neurons and two Dense layers, one with 25 neurons and the other with one neuron.
6. Compiling the Model
The LSTM model is compiled using the mean squared error (MSE) loss function and the adam optimizer.
7. Testing the model on testing data
The code below will get all the rows above the training_data_len from the column of the closing price. Then convert the x_test data set into the NumPy arrays so that they can be used to train the LSTM model.

As the LSTM model is expecting the data in 3-dimensional data set, using reshape() function we will reshape the data set in the form of 3-dimension.

Using the predict() function, get the predicted values from the model using the test data. And scaler.inverse_transform() function is undoing the scaling.
8. Error Calculation
RMSE is the root mean squared error, which helps to measure the accuracy of the model.
The lower the value, the better the model performs. The 0 value indicates the model’s predicted values match the actual values from the test data set perfectly.
rmse value we received was 0.6505512245089267 which is decent enough.

9. Make Predictions
The final step is to plot and visualize the data. To visualize the data we use these basic functions like title, label, plot as per how we want our graph to look like.
10. The Actual vs Predicted Values
Conclusion
Today we learned how to predict stock prices using an LSTM model! And the values for actual (close) and predicted (predictions) prices match quite a lot.

Thank you for reading!
