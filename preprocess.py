import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def convert_size_to_bhk(x):
    try:
        return int(str(x).split(' ')[0])
    except:
        return None

def convert_sqft(x):
    try:
        x = str(x)
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

def remove_outliers(df):
    df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

    df = df[df['total_sqft'] / df['bhk'] >= 300]

    def remove_pps_outliers(group):
        mean = group['price_per_sqft'].mean()
        std = group['price_per_sqft'].std()
        return group[
            (group['price_per_sqft'] >= mean - std) &
            (group['price_per_sqft'] <= mean + std)
        ]

    df = df.groupby('location', group_keys=False).apply(remove_pps_outliers)
    return df

def handle_location(df):
    df['location'] = df['location'].apply(lambda x: x.strip())

    location_stats = df['location'].value_counts()

    locations_less_than_10 = location_stats[location_stats <= 10]

    df['location'] = df['location'].apply(
        lambda x: 'other' if x in locations_less_than_10 else x
    )

    return df

def clean_data(df):
    df = df.drop(['society'], axis=1)

    df['bhk'] = df['size'].apply(convert_size_to_bhk)
    df = df.drop(['size'], axis=1)

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)

    df = df.dropna()

    df = remove_outliers(df)

    df = handle_location(df)

    return df