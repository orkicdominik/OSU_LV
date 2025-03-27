import sklearn
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('data_C02_emission.csv')

#  a)

numeric_features = ['Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)' , 'Fuel Consumption Comb (L/100km)' , 'Fuel Consumption Comb (mpg)']

X = data[numeric_features].to_numpy()
Y = data['CO2 Emissions (g/km)'].to_numpy()

X_train, X_test,y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

#  b)
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, color='blue', label='Skup za učenje')
plt.scatter(X_test[:, 0], y_test, color='red', label='Skup za testiranje')
plt.xlabel('Numerička varijabla')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()



#  c)
scl=MinMaxScaler()
 
X_train_scaled=scl.fit_transform(X_train)
X_test_scaled=scl.transform(X_test)
 
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(X_train[:, 0], bins=20, color='blue', edgecolor = 'black')
plt.title('Prije skaliranja')

plt.subplot(1, 2, 2)
plt.hist(X_train_scaled[:, 0], bins=20, color='blue', edgecolor = 'black')
plt.title('Nakon skaliranja')

plt.tight_layout()
plt.show()


#  d)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Koeficijenti modela:", model.coef_)



#  e)
y_pred = model.predict(X_test_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Stvarne vrijednosti')
plt.ylabel('Procijenjene vrijednosti')
plt.show()


#  f)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error: {mse}, Mean aboslute error: {mae}, R2: {r2}")
