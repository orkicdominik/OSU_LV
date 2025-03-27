import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv('data_C02_emission.csv')


encoder = OneHotEncoder()
encoder_dataframe = pd.DataFrame(encoder.fit_transform(data[['Fuel Type']]).toarray())
data = data.join(encoder_dataframe)


data.columns = ['Make', 'Model', 'Vehicle Class', 'Engine Size (L)', 'Cylinders', 'Transmission', 'Fuel Type', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'CO2 Emissions (g/km)', 'Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']


X = data.drop('CO2 Emissions (g/km)', axis=1)
y = data['CO2 Emissions (g/km)'].copy()



X_train_all, X_test_all, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


X_train = X_train_all[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]
X_test = X_test_all[['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Consumption Comb (mpg)', 'Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]


linearModel = LinearRegression()
linearModel.fit(X_train, y_train)


y_test_p = linearModel.predict(X_test)


plt.figure(figsize=(10, 6))
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, c='b', label='Stvarne vrijednosti', s=5, alpha=0.5)
plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test_p, c='r', label='Procijenjene vrijednosti', s=5, alpha=0.5)
plt.xlabel('Potrošnja goriva u gradu (L/100km)')
plt.ylabel('Emisija CO2 (g/km)')
plt.legend()
plt.show()


max_error_val = max_error(y_test, y_test_p)
print(f'Maksimalna pogreška: {max_error_val} g/km')


max_error_idx = np.argmax(np.abs(y_test - y_test_p))
print(f'Model vozila s maksimalnom pogreškom: {X_test_all.iloc[max_error_idx]["Model"]}')
