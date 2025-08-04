
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Ex 3.csv")
print("Dataset preview:")
print(data.head())

x=data[["age","income"]]
y=data["bought"]
print(x)
print(y)

#splitting data for training and testing
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte = train_test_split(x,y,test_size=0.3,random_state=0)

#fitting the logistic regression model to the training dtatset
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(xtr,ytr)

#prediction of the test and training set result
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
tepr=LR.predict(xte)
print("Prediction:",tepr.tolist())
accuracy=accuracy_score(yte,tepr)
print("Accuracy",accuracy)



#confusion matrix
cm=confusion_matrix(yte,tepr)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Not Bought (0)", "Bought (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Accuracy = {accuracy:.2f})")
plt.show()
