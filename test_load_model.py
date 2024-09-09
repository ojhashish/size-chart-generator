import joblib

try:
    # Adjust the path if necessary
    model = joblib.load(
        '/Users/aditya/Desktop/size_chart_generator/models/rf_model_cluster_0.pkl')
    print("Model loaded successfully")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except EOFError as e:
    print(f"EOFError: {e}")
except joblib.externals.loky.process_executor.TimeoutError as e:
    print(f"TimeoutError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
