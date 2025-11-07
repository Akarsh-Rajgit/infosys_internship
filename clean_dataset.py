import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv("Makaan_Properties_Buy.csv", encoding='latin1')

# 2. Inspect
print("Initial shape:", df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum().sort_values(ascending=False))

# 3. Drop columns with >60% missing
threshold = 0.25
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df = df.drop(columns=cols_to_drop)
print("Dropped columns:", list(cols_to_drop))

# 4. Drop rows with nulls in essential fields
essential = ["Price", "Size", "City_name"]
df = df.dropna(subset=essential)

# 5. Convert price strings to numbers
def parse_price(x):
    try:
        x = str(x).strip()
        if 'Cr' in x:
            return float(x.replace('Cr','').replace(',','').strip()) * 1e7
        elif 'Lac' in x or 'Lakh' in x or 'Lacs' in x:
            return float(x.replace('Lac','').replace('Lakh','').replace('Lacs','').replace(',','').strip()) * 1e5
        else:
            return float(x.replace(',','').strip())
    except:
        return np.nan

df['price_num'] = df['Price'].apply(parse_price)

# 6. Convert area to numbers
def parse_area(x):
    try:
        s = str(x)
        num = ''.join(ch for ch in s if ch.isdigit() or ch in ['.', ','])
        return float(num.replace(',',''))
    except:
        return np.nan

df['area_num'] = df['Size'].apply(parse_area)

# 7. Drop duplicates
df = df.drop_duplicates()

# 8. Handle outliers (remove top & bottom 1%)
for col in ['price_num', 'area_num']:
    q_low, q_high = df[col].quantile([0.01, 0.99])
    df = df[(df[col] >= q_low) & (df[col] <= q_high)]

# 9. Feature engineering
df['price_per_sqft'] = df['price_num'] / df['area_num']

# 10. Save cleaned dataset

df.to_csv("cleaned_indian_property.csv", index=False)
print("Cleaned dataset saved:", df.shape)
