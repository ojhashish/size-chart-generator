from sklearn.cluster import KMeans
import joblib


def perform_clustering(df, n_clusters=5):
    features = df[['Height', 'Weight', 'Chest',
                   'Waist', 'Hips', 'Body Shape Index']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
    joblib.dump(kmeans, 'kmeans_model.pkl')
    return df
