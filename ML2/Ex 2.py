import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('ex2.csv')
print("Dataset preview:")
print(data.head())
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=1/3,random_state=0)
#Fitting the Simple Linear Regression model to the training dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtr,ytr)
#Prediction of Test and Training set result
tepr=regressor.predict(xte)
trpr=regressor.predict(xtr)

from sklearn import metrics
print("MAE", metrics.mean_absolute_error(yte,tepr))
print("MSE", metrics.mean_squared_error(yte,tepr))
print("RMSE", nm.sqrt(metrics.mean_squared_error(yte,tepr)))


#Plot 1: Training Set (Actual vs Predicted)
plt.scatter(ytr, trpr, color="blue", label="Train")
plt.plot([min(ytr), max(ytr)], [min(ytr), max(ytr)], color='black', linestyle='--')
plt.title("Training Set: Actual vs Predicted Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.grid(True)
plt.show()

#Plot 2: Test Set (Actual vs Predicted)
plt.scatter(yte, tepr, color="red", label="Test")
plt.plot([min(yte), max(yte)], [min(yte), max(yte)], color='black', linestyle='--')
plt.title("Test Set: Actual vs Predicted Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.grid(True)
plt.show()

print(regressor.predict([[1300,3]]))
