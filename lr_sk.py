from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/mubaris/potential-enigma/master/headbrain.csv")

# Based on this blog https://mubaris.com/posts/linear-regression/

#Linear regression with scikit-learn

x = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values

m = len(x)

#Cannot use rank 1 matrix in scikit-learn
x = x.reshape((m, 1))

#create the model
reg = LinearRegression()

#Fit the data
reg = reg.fit(x,y)

#Prediction
y_pred = reg.predict(x)

#Obtain rmse and r2
mse = mean_squared_error(y,y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(x,y)

print(rmse)
print(r2_score)

