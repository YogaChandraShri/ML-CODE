import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data preview
data=pd.read_csv("ex1.csv")
print("dataset preview")
print(data.head())
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
print(x)
print(y)

#splitting data for training and testing
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=1/3,random_state=0)

#fitting the simple linear regression model to training dataset
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(xtr,ytr)

#prediction of the test and training set result
tepr=LR.predict(xte)
trpr=LR.predict(xtr)


from sklearn import metrics
print("MAE",metrics.mean_absolute_error(yte,tepr))
print("MSE",metrics.mean_squared_error(yte,tepr))
print("RSME",np.sqrt(metrics.mean_squared_error(yte,tepr)))

#plot 1:training set(showing model fit)
plt.scatter(xtr,ytr,color="green",label="training_data")
plt.plot(xtr,trpr,color="red",label="Regression Line")
plt.title("Training set:Area vs Price")
plt.xlabel("area")
plt.ylabel("price")
plt.legend()
plt.grid(True)
plt.show()

#plot 2
plt.scatter(xte,yte,color="green",label="test_data")
plt.plot(xte,tepr,color="red",label="predicted Line")
plt.title("Testing set:Area vs Price")
plt.xlabel("area")
plt.ylabel("price")
plt.legend()
plt.grid(True)
plt.show()

print("The predicted value for Area 1800 is",LR.predict([[1800]]))
