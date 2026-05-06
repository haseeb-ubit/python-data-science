import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create student data using NumPy
names = ["Ali", "Sara", "Ahmed", "Ayesha", "Bilal", "Zara"]
math = np.array([78, 85, 90, 66, 74, 88])
english = np.array([82, 79, 88, 70, 69, 92])
science = np.array([80, 91, 84, 73, 77, 89])

# Step 2: Create a Pandas DataFrame
data = {
    "Name": names,
    "Math": math,
    "English": english,
    "Science": science
}

df = pd.DataFrame(data)

# Step 3: Add a new column "Total" and "Average"
df["Total"] = df["Math"] + df["English"] + df["Science"]
df["Average"] = df["Total"] / 3

print("Student Data:\n")
print(df)

# Step 4: Find Topper
topper = df.loc[df["Total"].idxmax()]
print("\nTopper Student:")
print(topper)

# Step 5: Plot graph of marks
plt.plot(df["Name"], df["Math"])
plt.plot(df["Name"], df["English"])
plt.plot(df["Name"], df["Science"])

plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Student Marks in Subjects")
plt.legend(["Math", "English", "Science"])

plt.show()


# -------------------------------------------------------------------

import numpy as np

arr = np.array([1, 2, 4, 7, 8, 0, 5, 10, 9, 6, 3])
print("Total", arr.sum())
print("Max", arr.max())
print("Min", arr.min())
print("Average", arr.mean())


# -------------------------------------------------------------------

import numpy as np

arr = np.array([
    [1, 2, 3],
    [7, 8, 9],
    [5, 9, 0]
])
print("row sum", arr.sum(axis=1))
print("column sum", arr.sum(axis=0))
print("Average", arr.mean())


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

arr = np.array([89, 56, 34, 23, 12, 89, 90, 56, 43, 78, 23, 76, 75, 29])
mean = arr.mean()
median = np.median(arr)
variance = arr.var()
standard_deviation = arr.std()

print("Mean", mean)
print("Median", median)
print("Variance", variance)
print("Standard_deviation", standard_deviation)

plt.hist(arr, bins=5)
plt.xlabel("marks range")
plt.ylabel("students")
plt.title("records")
plt.show()


# -------------------------------------------------------------------

import matplotlib.pyplot as plt

categories = (["rent", "food", "transport", "lesure", "shopping"])
amounts = ([40000, 20000, 10000, 25000, 15000])

plt.figure()
plt.pie(amounts, labels=categories, autopct="%1.1f%%")
plt.title("monthly expenses")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Marks of 20 students
marks = np.array([95, 82, 67, 74, 88, 91, 53, 60, 77, 85,
                  69, 72, 58, 93, 81, 47, 66, 79, 84, 90])

# Function to assign grades
def get_grade(mark):
    if mark >= 85:
        return 'A'
    elif mark >= 70:
        return 'B'
    elif mark >= 60:
        return 'C'
    elif mark >= 50:
        return 'D'
    else:
        return 'F'

# Convert marks to grades
grades = np.array([get_grade(m) for m in marks])

# Count each grade
unique_grades, counts = np.unique(grades, return_counts=True)

print("Grades:", unique_grades)
print("Counts:", counts)

# Pie chart
plt.pie(counts, labels=unique_grades, autopct='%1.1f%%')
plt.title("Grade Distribution of Class")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

classes = ([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)])
frequency = ([4, 5, 7, 2, 9, 8])

midpoints = np.array([(a + b) / 2 for a, b in classes])
mean = np.sum(frequency * midpoints) / np.sum(frequency)
variance = np.sum(frequency * (midpoints - mean) ** 2) / np.sum(frequency)
standard_deviation = np.sqrt(variance)

print("mean", mean)
print("variance", variance)
print("standard_deviation", standard_deviation)

colors = ["red", "yellow", "blue", "green", "orange", "purple"]
class_starts = [c[0] for c in classes]
class_width = classes[0][1] - classes[0][0]

plt.bar(class_starts, frequency, width=class_width, align='edge', color=colors)
plt.xlabel("Class Intervals")
plt.ylabel("Frequeny")
plt.title("Histogram of Grouped Data")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
marks = np.array([40, 50, 60, 65, 70, 80])
model = LinearRegression()
model.fit(hours, marks)
predicted = model.predict([[10]])
print("predicted marks for 10 hours study :", predicted[0])

plt.figure()
plt.scatter(hours, marks)
plt.plot(hours, model.predict(hours))
plt.xlabel("study hours")
plt.ylabel("marks")
plt.title("record in regression model")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

size = np.array([500, 700, 900, 1100, 1300, 1500, 1700, 1900]).reshape(-1, 1)
price = np.array([50, 65, 70, 80, 85, 90, 100, 110])
model = LinearRegression()
model.fit(size, price)
predicted_price = model.predict([[1000]])
print("The amount for 1000 sq yards house is:", predicted_price[0], "thousand rupee")

plt.figure()
plt.scatter(size, price)
plt.plot(size, model.predict(size))
plt.xlabel("size of the house")
plt.ylabel("price of the house")
plt.title("prices of houses on their sizes")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ads_budget = np.array([20, 30, 40, 50, 60, 70]).reshape(-1, 1)
sales = np.array([15, 25, 33, 45, 48, 50])
model = LinearRegression()
model.fit(ads_budget, sales)
predicted = model.predict([[65]])
print("the predicted sales for 65k budget is", predicted, "thousand units")

plt.figure()
plt.scatter(ads_budget, sales)
plt.plot(ads_budget, model.predict(ads_budget))
plt.xlabel("Ads budget")
plt.ylabel("sales")
plt.title('record')
plt.show()


# -------------------------------------------------------------------

import matplotlib.pyplot as plt

categories = (["food", "rent", "transport", "lesure", "saving", "shopping"])
amount = ([15000, 30000, 5000, 10000, 10000, 7000])
plt.figure()
plt.pie(amount, labels=categories, autopct="%1.1f%%")
plt.title("monthly expenses chart")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

classes = [(0, 10), (20, 30), (40, 50), (60, 70), (80, 90), (100, 110)]
frq = ([2, 7, 4, 8, 9, 5])
midpoint = np.array([(a + b) / 2 for a, b in classes])
mean = np.sum(frq * midpoint) / np.sum(frq)
variance = np.sum(frq * (midpoint - mean) ** 2) / np.sum(frq)
standard_deviation = np.sqrt(variance)
print("mean", round(mean, 1))
print("variance", round(variance, 2))
print("stndard_deviation", round(standard_deviation, 2))


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

data = np.array([90, 90, 98, 78, 76, 45, 34, 89, 86, 75, 65, 78, 99, 91, 93, 65, 87, 88, 83, 82])
print("mean", np.mean(data))
print("median", np.median(data))
print("standard deviation", np.std(data))

plt.figure()
plt.hist(data, bins=3)
plt.xlabel("data")
plt.ylabel("data")
plt.title("histogram")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Temperature of 30 days
temps = np.array([
    30, 32, 31, 29, 35, 36, 34, 33, 32, 31,
    30, 29, 28, 27, 33, 34, 35, 36, 37, 38,
    36, 35, 34, 33, 32, 31, 30, 29, 28, 27
])

days = np.arange(1, 31)

mean_temp = np.mean(temps)
max_temp = np.max(temps)
min_temp = np.min(temps)

hottest_day = np.argmax(temps) + 1
coldest_day = np.argmin(temps) + 1

print(f"Mean Temperature: {mean_temp:.1f}")
print(f"Hottest Temperature: {max_temp} on Day {hottest_day}")
print(f"Coldest Temperature: {min_temp} on Day {coldest_day}")

plt.plot(days, temps)
plt.xlabel("Days")
plt.ylabel("Temperature (C)")
plt.title("Temperature of 30 Days")
plt.show()


# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

year = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]).reshape(-1, 1)
population = np.array([14.65, 15.02, 15.40, 15.74, 16.09, 16.46, 16.84, 17.23, 17.65, 18.08])
model = LinearRegression()
model.fit(year, population)
prediction = model.predict([[2026]])
print("The population of Karachi will be", round(prediction[0], 2), "million by 2026")

year_flat = year.flatten()

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.scatter(year_flat, population)
plt.plot(year_flat, model.predict(year))
plt.xlabel("year")
plt.ylabel("Population")
plt.title("regression prediction model")

plt.subplot(2, 2, 2)
plt.bar(year_flat, population)
plt.xlabel("year")
plt.ylabel("Population")
plt.title("bar graph")

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------

import numpy as np

arr = np.array([
    [1, 5, 8],
    [4, 0, 6],
    [2, 3, 7],
    [1, 6, 9]
])

sum = np.sum(arr, axis=1)
sum2 = np.sum(arr, axis=0)
average = np.mean(arr)

print("Row Sum:", sum)
print("Column Sum:")
for val in sum2:
    print(val)
print("Average:", round(average, 2))


# -------------------------------------------------------------------

import numpy as np

arr = np.array([
    [1, 5, 8],
    [4, 0, 6],
    [2, 3, 7]
])

det = np.linalg.det(arr)
print(round(det, 2))


# -------------------------------------------------------------------

import pandas as pd

data = {
    "Name": ["Haseeb", "Haiqa", "Jafar", "Kareen", "Minahil"],
    "math": [90, 90, 98, 92, 95],
    "science": [89, 86, 78, 75, 70],
    "english": [70, 75, 80, 80, 81],
}

df = pd.DataFrame(data)
print(df)
print("average\n")
print(df.mean(numeric_only=True))


# -------------------------------------------------------------------

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print("Mean:", np.mean(arr))
print("standard_deviation:", np.std(arr))
print("sum:", np.sum(arr))


# -------------------------------------------------------------------

import matplotlib.pyplot as plt

categories = (["rent", "groceries", "shopping", "savings", "lesure", "bills"])
amount = ([20000, 10000, 5000, 5000, 10000, 15000])

plt.figure()
plt.pie(amount, labels=categories, autopct="%1.1f%%")
plt.title("monthly expenses")
plt.show()


# -------------------------------------------------------------------
# Odoo - these only run inside an Odoo environment

# from odoo.exceptions import UserError

# def action_cancel(self):
#     for record in self:
#         if record.state == "sold":
#             raise UserError("Sold property cannot be canceled!")
#         record.state = "canceled"
#     return True


# def action_sold(self):
#     for record in self:
#         if record.state == "canceled":
#             raise UserError("Canceled property cannot be sold!")
#         record.state = "sold"
#     return True


# from odoo import models, fields, api

# class Student(models.Model):
#     _name = "school.student"
#     _description = "Student"

#     name = fields.Char(required=True)
#     student_class = fields.Char(string="Class")
#     marks = fields.Float()
#     result = fields.Selection([
#         ('pass', 'Pass'),
#         ('fail', 'Fail')
#     ], compute="_compute_result", store=True)

#     @api.depends("marks")
#     def _compute_result(self):
#         for rec in self:
#             rec.result = 'pass' if rec.marks >= 50 else 'fail'

#     def action_pass(self):
#         for rec in self:
#             rec.result = 'pass'
