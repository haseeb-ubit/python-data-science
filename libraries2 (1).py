import numpy as np

array = np.array([2, 5, 4, 12, 89, 0, 6, 5, 56])
print("mean:", np.mean(array))


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

classes = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
frq = np.array([12, 10, 5, 8, 7, 9])
midpoint = np.array([(a + b) / 2 for a, b in classes])
mean = np.sum(frq * midpoint) / np.sum(frq)
variance = np.sum(frq * (midpoint - mean) ** 2) / np.sum(frq)
standard_deviation = np.sqrt(variance)

print("mean", round(mean, 2))
print("variance", round(variance, 2))
print("standard deviation", round(standard_deviation, 2))

lables = [f"{a}-{b}" for a, b in classes]

plt.figure()
plt.bar(lables, frq)
plt.xlabel("classes")
plt.ylabel("frq")
plt.title("histogram")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

classes = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
frq = np.array([12, 10, 5, 8, 7, 9])

labels = [f"{a}-{b}" for a, b in classes]

plt.figure()

# Bar graph
plt.bar(labels, frq)

# Line graph on same data
plt.plot(labels, frq, marker='o')

plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.title("Bar and Line Graph")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
marks = np.array([30, 45, 50, 60, 65, 77, 80, 85])
model = LinearRegression()
model.fit(hours, marks)
predicted = model.predict([[10]])
print("the predicted marks will be:", predicted[0])

plt.figure()
plt.scatter(hours, marks)
plt.plot(hours, model.predict(hours))
plt.xlabel("hours")
plt.ylabel("marks")
plt.title("record in regression model")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

years = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]).reshape(-1, 1)
population = np.array([5, 7, 6, 5, 9, 10, 11, 12, 13, 18])
model = LinearRegression()
model.fit(years, population)
predicted = model.predict([[2026]])
print("the population in 2026 will be:", round(predicted[0], 2))

plt.figure()
plt.scatter(years, population)
plt.plot(years, model.predict(years))
plt.xlabel("years")
plt.ylabel("population")
plt.title("record in regression analysis")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

vehicles = np.array(["bus", "car", "bike", "auto", "metro"])
users = np.array([30, 15, 25, 20, 35])
colors = np.array(["red", "orange", "pink", "green", "blue"])

plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.bar(vehicles, users, color=colors)
plt.xlabel("vehicles")
plt.ylabel("users")
plt.title("no of people using different modes of transpoartation")

plt.subplot(2, 2, 2)
plt.pie(users, labels=vehicles, autopct="%1.1f%%")
plt.title("no of people using different modes of transpoartation")

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------

import numpy as np

array = np.array([
    [2, 3, 6],
    [6, 3, 0],
    [7, 8, 9]
])

print("row sum=", array.sum(axis=1))
print("column sum=", array.sum(axis=0))
print("average=", array.mean())


# -------------------------------------------------------------------

import numpy as np

A = np.array([
    [9, 8, 5],
    [6, 9, 5],
    [2, 3, 4]
])
B = np.array([
    [-9, 0, 7],
    [8, -1, 5],
    [5, 4, 3]
])
c = A / B
print(c)


# -------------------------------------------------------------------

import numpy as np

A = np.array([
    [4, 6, 0],
    [7, 3, 9],
    [1, 2, 5]
], dtype=float)
inv = np.linalg.inv(A)
print(inv)


# -------------------------------------------------------------------

import numpy as np

A = np.array([
    [8, 7, 6],
    [0, 1, 0],
    [5, 4, 3]
], dtype=float)
det = np.linalg.det(A)
if det == 0:
    print("matrix A is sigular and no inverse will be there")
else:
    print(det)
