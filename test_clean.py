from preprocess import load_data, clean_data

df = load_data("data/housing.csv")
df = clean_data(df)

print("Cleaned shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nUnique locations:", df['location'].nunique())
print("\nTop 10 locations:")
print(df['location'].value_counts().head(10))
print("\nFirst 5 rows:")
print(df.head())