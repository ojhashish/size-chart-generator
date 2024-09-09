# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import joblib
# import numpy as np

# # Load the data
# data = pd.read_csv(
#     '/Users/aditya/Desktop/size_chart_generator/data/body_measurements_dataset.csv')  # Updated path

# # Convert Height from feet and inches to inches


# def convert_height_to_inches(height_str):
#     if pd.isna(height_str):
#         return None
#     feet, inches = height_str.split("'")
#     inches = inches.replace('"', '')
#     return int(feet) * 12 + int(inches)


# data['Height'] = data['Height'].apply(convert_height_to_inches)

# # Handle missing values for Cup Size by filling with a placeholder value
# data['Cup Size'] = data['Cup Size'].fillna('None')

# # Convert categorical columns to numeric
# label_encoders = {}
# for column in ['Gender', 'Cup Size']:
#     le = LabelEncoder()
#     data[column] = le.fit_transform(data[column])
#     label_encoders[column] = le

# # Define features and target variable
# X = data[['Height', 'Weight', 'Bust/Chest', 'Waist',
#           'Hips', 'Gender', 'Cup Size', 'Body Shape Index']]
# y = data['Weight']  # Replace 'Weight' with your target variable

# # Assuming you have cluster labels for clustering
# # For demonstration, we'll use a placeholder list
# # Example: Replace with your actual cluster labels
# cluster_labels = np.random.randint(0, 5, size=len(data))

# # Check the length of cluster_labels
# assert len(cluster_labels) == len(
#     data), "Length of cluster_labels does not match length of data"

# # Train and save models for each cluster
# for i in range(5):  # Assuming you have 5 clusters
#     # Filter data for cluster i
#     cluster_mask = cluster_labels == i
#     X_cluster = X[cluster_mask]
#     y_cluster = y[cluster_mask]

#     # Ensure the cluster contains data
#     if X_cluster.empty or y_cluster.empty:
#         print(f"No data for cluster {i}, skipping...")
#         continue

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_cluster, y_cluster, test_size=0.2, random_state=42)

#     # Train the model
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)

#     # Save the model
#     # Updated path
#     joblib.dump(
#         rf_model, f'/Users/aditya/Desktop/size_chart_generator/models/rf_model_cluster_{i}.pkl')
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # Load the data from CSV
# df = pd.read_csv(
#     '/Users/aditya/Desktop/size_chart_generator/data/body_measurements_dataset.csv')

# # Convert Height to inches


# def height_to_inches(height):
#     try:
#         feet, inches = map(int, height.replace(
#             '\'', ' ').replace('\"', '').split())
#         return feet * 12 + inches
#     except:
#         return None  # Handle any conversion errors


# df['Height'] = df['Height'].apply(height_to_inches)

# # Handle missing values in columns like 'Cup Size' and 'Bust/Chest'
# df['Cup Size'] = df['Cup Size'].fillna('')
# df['Bust/Chest'] = df['Bust/Chest'].fillna(df['Bust/Chest'].median())

# # Encode categorical variables
# label_encoders = {}
# for column in ['Gender', 'Cup Size']:
#     le = LabelEncoder()
#     df[column] = le.fit_transform(df[column].astype(str))
#     label_encoders[column] = le

# # Define size categories


# def categorize_size(row):
#     if row['Weight'] < 60:
#         return 'S'
#     elif row['Weight'] < 80:
#         return 'M'
#     elif row['Weight'] < 100:
#         return 'L'
#     else:
#         return 'XL'


# df['Size'] = df.apply(categorize_size, axis=1)

# # Drop columns not used for prediction
# X = df.drop(columns=['Size'])
# y = df['Size']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# # Standardize numerical features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train RandomForestClassifier
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# # Make predictions and evaluate
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# # Function to predict size for new user data


# def predict_size(user_data):
#     # Convert user data to DataFrame
#     user_df = pd.DataFrame([user_data])

#     # Convert height to inches
#     user_df['Height'] = user_df['Height'].apply(height_to_inches)

#     # Handle missing values
#     user_df['Cup Size'] = user_df['Cup Size'].fillna('')

#     # Encode categorical variables
#     for column in ['Gender', 'Cup Size']:
#         user_df[column] = label_encoders[column].transform(
#             user_df[column].astype(str))

#     # Ensure all columns present in the training data are in the new data
#     missing_cols = set(X.columns) - set(user_df.columns)
#     for col in missing_cols:
#         # Add missing columns with default value (0 or another appropriate value)
#         user_df[col] = 0

#     user_df = user_df[X.columns]  # Reorder columns to match training data

#     # Standardize numerical features
#     user_df = scaler.transform(user_df)

#     # Predict size
#     prediction = clf.predict(user_df)
#     return prediction[0]


# # Example user data
# new_user_data = {
#     'Gender': 'Male',
#     'Height': "5'10\"",
#     'Weight': 60,
#     'Bust/Chest': 26,
#     'Cup Size': '',
#     'Waist': 28,
#     'Hips': 32,
#     'Body Shape Index': 0  # Make sure to include all necessary columns
# }

# predicted_size = predict_size(new_user_data)
# print(f"Predicted Size: {predicted_size}")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Load the data from CSV
# df = pd.read_csv(
#     '/Users/aditya/Desktop/size_chart_generator/data/body_measurements_dataset.csv')

# # Convert Height to inches


# def height_to_inches(height):
#     try:
#         feet, inches = map(int, height.replace(
#             '\'', ' ').replace('\"', '').split())
#         return feet * 12 + inches
#     except:
#         return None  # Handle any conversion errors


# df['Height'] = df['Height'].apply(height_to_inches)

# # Handle missing values in columns like 'Cup Size' and 'Bust/Chest'
# df['Cup Size'] = df['Cup Size'].fillna('')
# df['Bust/Chest'] = df['Bust/Chest'].fillna(df['Bust/Chest'].median())

# # Encode categorical variables
# label_encoders = {}
# for column in ['Gender', 'Cup Size']:
#     le = LabelEncoder()
#     df[column] = le.fit_transform(df[column].astype(str))
#     label_encoders[column] = le

# # Define size categories


# def categorize_size(row):
#     if row['Weight'] < 60:
#         return 'S'
#     elif row['Weight'] < 80:
#         return 'M'
#     elif row['Weight'] < 100:
#         return 'L'
#     else:
#         return 'XL'


# df['Size'] = df.apply(categorize_size, axis=1)

# # Drop columns not used for prediction
# X = df.drop(columns=['Size'])
# y = df['Size']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# # Standardize numerical features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train RandomForestClassifier
# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# # Make predictions and evaluate
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# # Save the model and scaler
# joblib.dump(clf, 'size_predictor_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(label_encoders, 'label_encoders.pkl')

# # Function to predict size for new user data


# def predict_size(user_data):
#     # Convert user data to DataFrame
#     user_df = pd.DataFrame([user_data])

#     # Convert height to inches
#     user_df['Height'] = user_df['Height'].apply(height_to_inches)

#     # Handle missing values
#     user_df['Cup Size'] = user_df['Cup Size'].fillna('')

#     # Encode categorical variables
#     for column in ['Gender', 'Cup Size']:
#         user_df[column] = label_encoders[column].transform(
#             user_df[column].astype(str))

#     # Ensure all columns present in the training data are in the new data
#     missing_cols = set(X.columns) - set(user_df.columns)
#     for col in missing_cols:
#         # Add missing columns with default value (0 or another appropriate value)
#         user_df[col] = 0

#     user_df = user_df[X.columns]  # Reorder columns to match training data

#     # Standardize numerical features
#     user_df = scaler.transform(user_df)

#     # Predict size
#     prediction = clf.predict(user_df)
#     return prediction[0]

# # Get user input


# def get_user_input():
#     gender = input("Enter Gender (Male/Female): ")
#     height = input("Enter Height (e.g., 5'7\"): ")
#     weight = float(input("Enter Weight (in kg): "))
#     bust_chest = float(input("Enter Bust/Chest measurement (in cm): "))
#     cup_size = input("Enter Cup Size (e.g., A, B, C, D): ")
#     waist = float(input("Enter Waist measurement (in cm): "))
#     hips = float(input("Enter Hips measurement (in cm): "))
#     body_shape_index = int(input("Enter Body Shape Index (0-4): "))

#     return {
#         'Gender': gender,
#         'Height': height,
#         'Weight': weight,
#         'Bust/Chest': bust_chest,
#         'Cup Size': cup_size,
#         'Waist': waist,
#         'Hips': hips,
#         'Body Shape Index': body_shape_index
#     }


# # Predict size for new user input
# new_user_data = get_user_input()
# predicted_size = predict_size(new_user_data)
# print(f"Predicted Size: {predicted_size}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the data from CSV
df = pd.read_csv(
    '/Users/aditya/Desktop/size_chart_generator/data/body_measurements_dataset.csv')

# Convert Height from feet and inches to inches


def height_to_inches(height):
    try:
        feet, inches = map(int, height.replace(
            '\'', ' ').replace('\"', '').split())
        return feet * 12 + inches
    except:
        return None


df['Height'] = df['Height'].apply(height_to_inches)

# Handle missing values in columns like 'Cup Size' and 'Bust/Chest'
df['Cup Size'] = df['Cup Size'].fillna('')
df['Bust/Chest'] = df['Bust/Chest'].fillna(df['Bust/Chest'].median())

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'Cup Size']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Define size categories with XXL


def categorize_size(row):
    if row['Weight'] < 60:
        return 'S'
    elif row['Weight'] < 80:
        return 'M'
    elif row['Weight'] < 100:
        return 'L'
    elif row['Weight'] < 120:
        return 'XL'
    else:
        return 'XXL'


df['Size'] = df.apply(categorize_size, axis=1)

# Drop columns not used for prediction
X = df.drop(columns=['Size'])
y = df['Size']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForestClassifier with more estimators to improve accuracy
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model, scaler, and label encoders
joblib.dump(clf, 'size_predictor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Function to predict size for new user data with confidence scores


def predict_size_with_confidence(user_data):
    # Convert user data to DataFrame
    user_df = pd.DataFrame([user_data])

    # Convert height to inches
    user_df['Height'] = user_df['Height'].apply(height_to_inches)

    # Handle missing values
    user_df['Cup Size'] = user_df['Cup Size'].fillna('')

    # Encode categorical variables
    for column in ['Gender', 'Cup Size']:
        user_df[column] = label_encoders[column].transform(
            user_df[column].astype(str))

    # Ensure all columns present in the training data are in the new data
    missing_cols = set(X.columns) - set(user_df.columns)
    for col in missing_cols:
        # Add missing columns with default value (0 or another appropriate value)
        user_df[col] = 0

    user_df = user_df[X.columns]  # Reorder columns to match training data

    # Standardize numerical features
    user_df = scaler.transform(user_df)

    # Predict size
    prediction = clf.predict(user_df)
    prediction_proba = clf.predict_proba(user_df)

    # Get the confidence score for the predicted size
    predicted_size = prediction[0]
    confidence_score = max(prediction_proba[0])

    return predicted_size, confidence_score

# Get user input in inches


def get_user_input():
    gender = input("Enter Gender (Male/Female): ")
    height = input("Enter Height (in inches, e.g., 67 for 5'7\"): ")
    weight = float(input("Enter Weight (in kg): "))
    bust_chest = float(input("Enter Bust/Chest measurement (in inches): "))
    cup_size = input("Enter Cup Size (e.g., A, B, C, D): ")
    waist = float(input("Enter Waist measurement (in inches): "))
    hips = float(input("Enter Hips measurement (in inches): "))
    body_shape_index = int(input("Enter Body Shape Index (0-4): "))

    return {
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'Bust/Chest': bust_chest,
        'Cup Size': cup_size,
        'Waist': waist,
        'Hips': hips,
        'Body Shape Index': body_shape_index
    }


# Predict size for new user input and get confidence score
new_user_data = get_user_input()
predicted_size, confidence_score = predict_size_with_confidence(new_user_data)
print(f"Predicted Size: {predicted_size}")
print(f"Confidence Score: {confidence_score:.2f}")

# Display size chart with confidence scores (Example values)
size_chart = {
    'Size': ['S', 'M', 'L', 'XL', 'XXL'],
    'Chest Size (inches)': [34, 36, 38, 40, 42],  # Example values
    'Waist Size (inches)': [28, 30, 32, 34, 36],  # Example values
    # Placeholder example values
    'Confidence Score': [0.95, 0.92, 0.93, 0.90, 0.88]
}

size_chart_df = pd.DataFrame(size_chart)
print("\nSize Chart:")
print(size_chart_df)
