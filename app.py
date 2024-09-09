# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the KMeans model and RandomForest models
# try:
#     kmeans = joblib.load(
#         '/Users/aditya/Desktop/size_chart_generator/models/kmeans_model.pkl')
#     models = {i: joblib.load(
#         f'/Users/aditya/Desktop/size_chart_generator/models/rf_model_cluster_{i}.pkl') for i in range(5)}
#     print("Models loaded successfully.")
# except Exception as e:
#     print(f"Error loading models: {e}")


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         df = pd.DataFrame([data])

#         # Convert height from feet and inches to inches
#         def convert_height_to_inches(height_str):
#             if pd.isna(height_str):
#                 return None
#             feet, inches = height_str.split("'")
#             inches = inches.replace('"', '')
#             return int(feet) * 12 + int(inches)

#         df['Height'] = df['Height'].apply(convert_height_to_inches)

#         # Convert categorical columns to numeric
#         label_encoders = {}
#         for column in ['Gender', 'Cup Size']:
#             if column in df.columns:
#                 le = LabelEncoder()
#                 df[column] = le.fit_transform(df[column])
#                 label_encoders[column] = le

#         # Features for clustering
#         features = ['Height', 'Weight', 'Bust/Chest', 'Waist',
#                     'Hips', 'Gender', 'Cup Size', 'Body Shape Index']
#         X = df[features]

#         # Standardize features
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         # Predict cluster
#         cluster = kmeans.predict(X_scaled)[0]
#         model = models[cluster]

#         # Predict size
#         prediction = model.predict(X_scaled)[0]

#         return jsonify({'size_prediction': prediction})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# if __name__ == '__main__':
#     app.run(debug=True, port=3000)
# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# import joblib

# # Initialize Flask app
# app = Flask(__name__)

# # Load the saved model, scaler, and label encoders
# model = joblib.load('size_predictor_model.pkl')
# scaler = joblib.load('scaler.pkl')
# label_encoders = joblib.load('label_encoders.pkl')

# # Define the columns used during model training
# columns = [
#     'Gender', 'Height', 'Weight', 'Bust/Chest', 'Cup Size',
#     'Waist', 'Hips', 'Body Shape Index'
# ]

# # Helper functions


# def height_to_inches(height):
#     try:
#         feet, inches = map(int, height.replace(
#             '\'', ' ').replace('\"', '').split())
#         return feet * 12 + inches
#     except:
#         return None


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
#     for col in columns:
#         if col not in user_df.columns:
#             # Add missing columns with default value (0 or another appropriate value)
#             user_df[col] = 0

#     user_df = user_df[columns]  # Reorder columns to match training data

#     # Standardize numerical features
#     user_df = scaler.transform(user_df)

#     # Predict size
#     prediction = model.predict(user_df)
#     return prediction[0]

# # Define route for index page


# @app.route('/')
# def index():
#     return render_template('index.html')

# # Define route to handle form submission and prediction


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get input data from the form
#     user_data = {
#         'Gender': request.form['Gender'],
#         'Height': request.form['Height'],
#         'Weight': float(request.form['Weight']),
#         'Bust/Chest': float(request.form['Bust/Chest']),
#         'Cup Size': request.form['Cup Size'],
#         'Waist': float(request.form['Waist']),
#         'Hips': float(request.form['Hips']),
#         'Body Shape Index': int(request.form['Body Shape Index'])
#     }

#     # Predict size
#     predicted_size = predict_size(user_data)

#     # Return the result as JSON
#     return jsonify({'Predicted Size': predicted_size})


# # Run the app on a custom port (e.g., 8080)
# if __name__ == '__main__':
#     app.run(debug=True, port=8080)
# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# import joblib

# # Initialize Flask app
# app = Flask(__name__)

# # Load the saved models, scaler, and label encoders
# model = joblib.load('size_predictor_model.pkl')
# scaler = joblib.load('scaler.pkl')
# label_encoders = joblib.load('label_encoders.pkl')

# # Define the columns used during model training
# columns = [
#     'Gender', 'Height', 'Weight', 'Bust/Chest', 'Cup Size',
#     'Waist', 'Hips', 'Body Shape Index'
# ]

# # Helper functions


# def height_to_inches(height):
#     try:
#         feet, inches = map(int, height.replace(
#             '\'', ' ').replace('\"', '').split())
#         return feet * 12 + inches
#     except:
#         return None


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
#     for col in columns:
#         if col not in user_df.columns:
#             user_df[col] = 0  # Add missing columns with default value

#     user_df = user_df[columns]  # Reorder columns to match training data

#     # Standardize numerical features
#     user_df = scaler.transform(user_df)

#     # Predict size
#     predicted_size = model.predict(user_df)[0]

#     # Example: Generate confidence score and size chart
#     confidence_score = 90  # Placeholder; implement based on your model's output
#     size_chart = [
#         {"Size": "S", "Chest_Size": 34, "Waist_Size": 28},
#         {"Size": "M", "Chest_Size": 38, "Waist_Size": 32},
#         {"Size": "L", "Chest_Size": 42, "Waist_Size": 36},
#         {"Size": "XL", "Chest_Size": 46, "Waist_Size": 40},
#         {"Size": "XXL", "Chest_Size": 50, "Waist_Size": 44}
#     ]

#     return predicted_size, confidence_score, size_chart

# # Define route for index page


# @app.route('/')
# def index():
#     return render_template('index.html')

# # Define route to handle form submission and prediction


# @app.route('/predict', methods=['post'])
# def predict():
#     try:
#         # Get input data from the form
#         user_data = {
#             'Gender': request.form['Gender'],
#             'Height': request.form['Height'],
#             'Weight': float(request.form['Weight']),
#             'Bust/Chest': float(request.form['Bust/Chest']),
#             'Cup Size': request.form['Cup Size'],
#             'Waist': float(request.form['Waist']),
#             'Hips': float(request.form['Hips']),
#             'Body Shape Index': int(request.form['Body Shape Index'])
#         }

#         # Predict size
#         predicted_size, confidence_score, size_chart = predict_size(user_data)

#         # Return the result as JSON
#         return jsonify({
#             'Predicted Size': predicted_size,
#             'Confidence Score': confidence_score,
#             'Size Chart': size_chart
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# # Run the app on a custom port (e.g., 8080)
# if __name__ == '__main__':
#     app.run(debug=True, port=8080)
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the saved model, scaler, and label encoders
model = joblib.load('size_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the columns used during model training
columns = [
    'Gender', 'Height', 'Weight', 'Bust/Chest', 'Cup Size',
    'Waist', 'Hips', 'Body Shape Index'
]

# Helper functions


def height_to_inches(height):
    try:
        feet, inches = map(int, height.replace(
            '\'', ' ').replace('\"', '').split())
        return feet * 12 + inches
    except:
        return None


def predict_size(user_data):
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
    for col in columns:
        if col not in user_df.columns:
            # Add missing columns with default value (0 or another appropriate value)
            user_df[col] = 0

    user_df = user_df[columns]  # Reorder columns to match training data

    # Standardize numerical features
    user_df = scaler.transform(user_df)

    # Predict size
    prediction = model.predict(user_df)
    return prediction[0]

# Define route for index page


@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle form submission and prediction


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    user_data = {
        'Gender': request.form['Gender'],
        'Height': request.form['Height'],
        'Weight': float(request.form['Weight']),
        'Bust/Chest': float(request.form['Bust/Chest']),
        'Cup Size': request.form['Cup Size'],
        'Waist': float(request.form['Waist']),
        'Hips': float(request.form['Hips']),
        'Body Shape Index': int(request.form['Body Shape Index'])
    }

    # Predict size
    predicted_size = predict_size(user_data)

    # Return the result as JSON
    return jsonify({'Predicted Size': predicted_size})


# Run the app on a custom port (e.g., 8080)
if __name__ == '__main__':
    app.run(debug=True, port=8080)
