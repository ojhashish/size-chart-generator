import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Cup Size'].fillna('NA', inplace=True)
    df['Body Shape Index'].fillna(-1, inplace=True)

    def height_to_inches(height_str):
        feet, inches = map(int, height_str.split('.'))
        return feet * 12 + inches
    df['Height'] = df['Height'].apply(height_to_inches)
    scaler = StandardScaler()
    df[['Height', 'Weight', 'Chest', 'Waist', 'Hips']] = scaler.fit_transform(
        df[['Height', 'Weight', 'Chest', 'Waist', 'Hips']])
    return df
