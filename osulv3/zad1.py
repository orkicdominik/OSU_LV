import pandas as pd

data = pd.read_csv('data_C02_emission.csv')


print("a)")
print(f"Number of measurements: {len(data)}")

data.info()

if data.isnull().values.any():
    data = data.dropna()
else:
    print("No null values")

if data.duplicated().values.any():
    data = data.drop_duplicates()
else:
    print("No duplicate values")


data["Make"] = data["Make"].astype("category")
data["Model"] = data["Model"].astype("category")
data["Vehicle Class"] = data["Vehicle Class"].astype("category")
data["Transmission"] = data["Transmission"].astype("category")
data["Fuel Type"] = data["Fuel Type"].astype("category")


data.info()

print("b)")

city_usage = data.sort_values("Fuel Consumption City (L/100km)")
print("Biggest spenders:")
print(city_usage[["Make", "Model", "Fuel Consumption City (L/100km)"]].tail(3))
print("Smallest spenders:")
print(city_usage[["Make", "Model", "Fuel Consumption City (L/100km)"]].head(3))


print("c)")

medium_motors = data[(data["Engine Size (L)"] > 2.5) & data["Engine Size (L)"] < 3.5]
print(f"{len(medium_motors)} Cars with medium sized motors: ")
print(f"Their average CO2 consumption: {medium_motors["CO2 Emissions (g/km)"].mean()} g/km")

print("d)")

audi = data[data["Make"] == "Audi"]
print(f"Number of Audis:{len(audi)}")
print(f"Average CO2 emissions of 4-cylinder audis: {audi[audi["Cylinders"] == 4]["CO2 Emissions (g/km)"].mean()}")


print("e)")

cylinders = data[(data["Cylinders"] > 2) & (data["Cylinders"] % 2 == 0)]
print(f"Number cars with more than 2 cylinders: {len(cylinders)}")
print("Their average emissions (g/km):")
print(cylinders.groupby("Cylinders")["CO2 Emissions (g/km)"].mean())

print("f)")

diesel = data[data["Fuel Type"] == "D"]
gasoline = data[(data["Fuel Type"] == "X") & (data["Fuel Type"] == "Z")]
print("Average city consumption for:")
print(f"\tdiesel cars: {diesel["Fuel Consumption City (L/100km)"].mean()} L/100km")
print(f"\tgasoline cars: {gasoline["Fuel Consumption City (L/100km)"].mean()} L/100km")
print("Median city consumption for:")
print(f"\tdiesel cars: {diesel["Fuel Consumption City (L/100km)"].median()} L/100km")
print(f"\tgasoline cars: {gasoline["Fuel Consumption City (L/100km)"].median()} L/100km")

