import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def run_clustering(df):

    df.columns = ["CustomerID", "Gender", "Age", "Income", "SpendingScore"]
    df = df.drop("CustomerID", axis=1)

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["SpendingPower"] = df["Income"] * df["SpendingScore"]

    features = ["Gender", "Age", "Income", "SpendingScore", "SpendingPower"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    df["PCA1"] = components[:, 0]
    df["PCA2"] = components[:, 1]

    return df