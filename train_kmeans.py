import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load your data
data = pd.read_csv(
    '/Users/aditya/Desktop/size_chart_generator/data/body_measurements_dataset.csv')

# Function to convert Height from feet and inches to inches


def convert_height_to_inches(height_str):
    if pd.isna(height_str):
        return None
    feet, inches = height_str.split("'")
    inches = inches.replace('"', '')
    return int(feet) * 12 + int(inches)


data['Height'] = data['Height'].apply(convert_height_to_inches)
data['Cup Size'] = data['Cup Size'].fillna('None')

# Convert categorical columns to numeric
label_encoders = {}
for column in ['Gender', 'Cup Size']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features for clustering
X = data[['Height', 'Weight', 'Bust/Chest', 'Waist',
          'Hips', 'Gender', 'Cup Size', 'Body Shape Index']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Save the KMeans model
# Update this path if necessary
joblib.dump(
    kmeans, '/Users/aditya/Desktop/size_chart_generator/models/kmeans_model.pkl')
