import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")


data.hist("CO2 Emissions (g/km)", bins=50, edgecolor = 'black')
plt.title("CO2 Emissions histogram")
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Number of cars")
plt.show()


fuel = data.groupby("Fuel Type")
for key, group in fuel:
    plt.scatter(group["Fuel Consumption City (L/100km)"], group["CO2 Emissions (g/km)"], label=key)
plt.title("Emissions to consumption")
plt.xlabel("City fuel consumption (L/100km)")
plt.ylabel("CO2 Emissions histogram g/km")
plt.legend()
plt.show()

data.boxplot("Fuel Consumption Hwy (L/100km)", by="Fuel Type")
plt.suptitle("")
plt.title("Out-of-city fuel consumption by fuel type")
plt.xlabel("Fuel type")
plt.ylabel("Fuel consumption (L/100km)")
plt.show()


fuel.size().plot(kind="bar")
plt.title("Number of cars by fuel type")
plt.xlabel("Fuel type")
plt.ylabel("Number of cars")
plt.show()


cylinders = data.groupby("Cylinders")
cylinders["CO2 Emissions (g/km)"].mean().plot(kind="bar")
plt.title("Average emissions by number of cylinders")
plt.xlabel("Number of cylinders")
plt.ylabel("Average emissions (g/km)")
plt.show()
