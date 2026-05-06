import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# =============================================================================
#  Student Performance Analysis System
#  A mini data science project to analyze student marks, visualize results,
#  and predict future performance using linear regression.
# =============================================================================


# -----------------------------------------------------------------------------
# 1. DATASET
# -----------------------------------------------------------------------------

data = {
    "Name": ["Ali", "Ubaid", "Maaz", "Laiba", "Zainab", "Faiza", "Haseeb", "Hafsa", "Bilal", "Sara"],
    "Math": [78, 76, 65, 75, 79, 71, 73, 72, 80, 68],
    "Science": [80, 85, 77, 89, 81, 79, 84, 83, 76, 74],
    "English": [70, 70, 73, 76, 71, 72, 75, 79, 69, 77],
    "Pakistan Studies": [65, 67, 62, 70, 72, 71, 69, 68, 66, 73],
    "Urdu": [70, 70, 71, 77, 67, 76, 66, 79, 72, 75],
}

df = pd.DataFrame(data)


# -----------------------------------------------------------------------------
# 2. CALCULATED COLUMNS
# -----------------------------------------------------------------------------

subjects = ["Math", "Science", "English", "Pakistan Studies", "Urdu"]

df["Total"] = df[subjects].sum(axis=1)
df["Average"] = (df["Total"] / len(subjects)).round(2)
df["Percentage"] = ((df["Total"] / (len(subjects) * 100)) * 100).round(2)


# grade based on percentage
def assign_grade(percentage):
    if percentage >= 85:
        return "A"
    elif percentage >= 70:
        return "B"
    elif percentage >= 60:
        return "C"
    elif percentage >= 50:
        return "D"
    else:
        return "F"

df["Grade"] = df["Percentage"].apply(assign_grade)


# -----------------------------------------------------------------------------
# 3. BASIC STATS
# -----------------------------------------------------------------------------

print("=" * 60)
print("         STUDENT PERFORMANCE ANALYSIS REPORT")
print("=" * 60)
print()
print(df[["Name", "Total", "Average", "Percentage", "Grade"]].to_string(index=False))
print()

print("-" * 40)
print("CLASS STATISTICS")
print("-" * 40)
print(f"Class Average  : {df['Average'].mean():.2f}")
print(f"Highest Score  : {df['Total'].max()} — {df.loc[df['Total'].idxmax(), 'Name']}")
print(f"Lowest Score   : {df['Total'].min()} — {df.loc[df['Total'].idxmin(), 'Name']}")
print(f"Std Deviation  : {df['Total'].std():.2f}")
print(f"Variance       : {df['Total'].var():.2f}")
print()

print("-" * 40)
print("TOP 3 STUDENTS")
print("-" * 40)
top3 = df.sort_values("Total", ascending=False).head(3)
for i, row in enumerate(top3.itertuples(), 1):
    print(f"  {i}. {row.Name} — Total: {row.Total}, Grade: {row.Grade}")
print()

print("-" * 40)
print("SUBJECT AVERAGES")
print("-" * 40)
for subject in subjects:
    print(f"  {subject:<20}: {df[subject].mean():.2f}")
print()


# -----------------------------------------------------------------------------
# 4. VISUALIZATIONS
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Student Performance Analysis", fontsize=16, fontweight="bold")


# --- Chart 1: Total Marks Bar Chart ---
colors = ["green" if g == "A" else "steelblue" if g == "B" else "orange" if g == "C" else "red"
          for g in df["Grade"]]

axes[0, 0].bar(df["Name"], df["Total"], color=colors, edgecolor="black", linewidth=0.5)
axes[0, 0].set_title("Total Marks per Student")
axes[0, 0].set_xlabel("Student")
axes[0, 0].set_ylabel("Total Marks")
axes[0, 0].set_ylim(0, 550)
axes[0, 0].tick_params(axis="x", rotation=45)

for i, (name, total) in enumerate(zip(df["Name"], df["Total"])):
    axes[0, 0].text(i, total + 5, str(total), ha="center", fontsize=8)


# --- Chart 2: Grade Distribution Pie Chart ---
grade_counts = df["Grade"].value_counts()
pie_colors = {"A": "green", "B": "steelblue", "C": "orange", "D": "red", "F": "gray"}
pie_color_list = [pie_colors.get(g, "gray") for g in grade_counts.index]

axes[0, 1].pie(grade_counts.values, labels=grade_counts.index, autopct="%1.1f%%",
               colors=pie_color_list, startangle=90)
axes[0, 1].set_title("Grade Distribution")


# --- Chart 3: Subject-wise Average Bar Chart ---
subject_avgs = df[subjects].mean()

axes[1, 0].bar(subjects, subject_avgs, color="steelblue", edgecolor="black", linewidth=0.5)
axes[1, 0].set_title("Subject-wise Class Average")
axes[1, 0].set_xlabel("Subject")
axes[1, 0].set_ylabel("Average Marks")
axes[1, 0].set_ylim(0, 100)
axes[1, 0].tick_params(axis="x", rotation=30)

for i, avg in enumerate(subject_avgs):
    axes[1, 0].text(i, avg + 1, f"{avg:.1f}", ha="center", fontsize=8)


# --- Chart 4: Line graph — individual subject performance ---
for subject in subjects:
    axes[1, 1].plot(df["Name"], df[subject], marker="o", label=subject, linewidth=1.5)

axes[1, 1].set_title("Subject-wise Performance per Student")
axes[1, 1].set_xlabel("Student")
axes[1, 1].set_ylabel("Marks")
axes[1, 1].legend(fontsize=7, loc="lower right")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("performance_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("Charts saved as performance_charts.png")
print()


# -----------------------------------------------------------------------------
# 5. LINEAR REGRESSION — Predicting Total from Study Hours
#    (simulated study hours data paired with actual totals)
# -----------------------------------------------------------------------------

study_hours = np.array([3, 4, 2, 5, 4, 3, 4, 4, 3, 3]).reshape(-1, 1)
total_marks = df["Total"].values

model = LinearRegression()
model.fit(study_hours, total_marks)

print("-" * 40)
print("REGRESSION — Study Hours vs Total Marks")
print("-" * 40)
print(f"  Slope (marks per hour) : {model.coef_[0]:.2f}")
print(f"  Intercept              : {model.intercept_:.2f}")

for hrs in [5, 6, 7, 8]:
    prediction = model.predict([[hrs]])[0]
    print(f"  Predicted total for {hrs}h study: {prediction:.1f}")
print()

# Regression plot
plt.figure(figsize=(7, 4))
plt.scatter(study_hours, total_marks, color="steelblue", zorder=5, label="Actual")
plt.plot(study_hours, model.predict(study_hours), color="red", linewidth=2, label="Regression Line")
plt.xlabel("Study Hours per Day")
plt.ylabel("Total Marks (out of 500)")
plt.title("Study Hours vs Total Marks — Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("regression_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Regression plot saved as regression_plot.png")
print()


# -----------------------------------------------------------------------------
# 6. PASS / FAIL SUMMARY
# -----------------------------------------------------------------------------

passed = df[df["Percentage"] >= 50]
failed = df[df["Percentage"] < 50]

print("-" * 40)
print("PASS / FAIL SUMMARY")
print("-" * 40)
print(f"  Passed : {len(passed)} students")
print(f"  Failed : {len(failed)} students")
print(f"  Pass Rate: {(len(passed)/len(df))*100:.0f}%")

if not failed.empty:
    print(f"\n  Students who need attention:")
    for name in failed["Name"]:
        print(f"    - {name}")

print()
print("=" * 60)
print("  Analysis Complete.")
print("=" * 60)
