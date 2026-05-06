# Python Data Science Practice

This repository contains Python practice files I worked on while learning data science libraries during my BS Software Engineering. The code covers NumPy, Pandas, Matplotlib and Scikit-learn — starting from basics and building up to a small project at the end.

---

## Files

### `libraries.py` — Data Science Practice 1
Basic to intermediate practice using NumPy, Pandas and Matplotlib. Things like array operations, statistics (mean, median, variance, standard deviation), DataFrames, and different chart types — bar charts, pie charts, histograms and line graphs. Also includes a few linear regression examples towards the end (study hours vs marks, house size vs price, ads budget vs sales) and some Karachi population prediction using regression. The Odoo bits at the bottom are kept commented out since those only run inside an Odoo environment.

### `libraries2.py` — Data Science Practice 2
More practice on the same topics — grouped data statistics, combined bar and line graphs, regression models, matrix operations (row/column sums, element-wise division, inverse, determinant). Pretty much continued from where the first file left off.

### `pandas_practice.py` — Data Science Practice 3
Focused specifically on Pandas. Creating a DataFrame from scratch, adding new rows and columns dynamically, calculating totals and averages across 5 subjects, and sorting to find the top 3 students. Simple but it covers the core Pandas workflow.

### `student_performance_analysis.py` — Mini Project
This one I put together by combining everything from the practice files into one proper script. It takes a dataset of 10 students with marks in 5 subjects (Math, Science, English, Pakistan Studies, Urdu), then:

- calculates total marks, average, percentage and grade for each student
- prints a full class report with stats like highest score, lowest score, standard deviation
- shows top 3 students
- generates 4 charts in one figure — total marks bar chart, grade distribution pie chart, subject averages and a subject-wise line graph per student
- runs a linear regression on study hours vs total marks and predicts scores for 5 to 8 hours of study
- saves both charts as PNG files

---

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## How to Run

Make sure you have the libraries installed:
pip install numpy pandas matplotlib scikit-learn

Then just run whichever file you want:
python student_performance_analysis.py

---

## Note

The practice files (`libraries.py`, `libraries2.py`) are meant to be run section by section — each block is independent. Running the whole file at once will work but it'll open multiple chart windows one after the other.
