import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("https://raw.githubusercontent.com/mubaris/potential-enigma/master/headbrain.csv")
#print(data.head())
#print(data.shape)

# Based on this blog https://mubaris.com/posts/linear-regression/

#Linear regression with OLS

x = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values

mean_x = np.mean(x)
mean_y = np.mean(y)

m = len(x)

#Linear regression with OLS

numerator = 0
denominator = 0

for i in range(m):
    numerator += (x[i] - mean_x) * (y[i] - mean_y)
    denominator += (x[i] - mean_x) ** 2

b1 = numerator / denominator
b0 = mean_y - (b1 * mean_x)

print(b1,b0)

#Lets plot these
max_x = np.max(x) + 100
min_x = np.min(x) - 100

x1 = np.linspace(min_x, max_x, 1000)
y1 = b0 + (b1 * x1)

plt.plot(x1,y1, color='#58b970', label='Regression line')
plt.scatter(x,y, c='#ef5423', label='Scatter plot')
plt.xlabel('Head size')
plt.ylabel('Brain weight')
plt.legend()
plt.show()

#Calculate the rmse

rmse = 0

for i in range(m):
    y_pred = b0 + (b1*x[i])
    rmse += (y[i] - y_pred)**2

rmse = np.sqrt(rmse/m)
print(rmse)

ss_t = 0
ss_r = 0

for i in range(m):
    y_pred = b0 + (b1 * x[i])
    ss_t += (y[i] - mean_y)**2
    ss_r += (y[i] - y_pred)**2

r2 = 1 - (ss_r/ss_t)
print(r2)