import pandas as pd

data = {
    "Names": ["Ali", "Ubaid", "Maaz", "Laiba", "Zainab", "Faiza"],
    "Math": [78, 76, 65, 75, 79, 71],
    "Science": [80, 85, 77, 89, 81, 79],
    "English": [70, 70, 73, 76, 71, 72]
}
df = pd.DataFrame(data)

df.loc[len(df)] = ["Haseeb", 73, 84, 75]
df.loc[len(df)] = ["Hafsa", 72, 83, 79]
df["Pakistan Studies"] = [65, 67, 62, 70, 72, 71, 69, 68]
df["Urdu"] = [70, 70, 71, 77, 67, 76, 66, 79]

df["total"] = df[["Math", "Science", "English", "Pakistan Studies", "Urdu"]].sum(axis=1)
df["Average"] = df["total"] / 5
print(df)

df_sorting = df.sort_values(by="total", ascending=False)
print(df_sorting[["Names", "total", "Average"]].head(3))
