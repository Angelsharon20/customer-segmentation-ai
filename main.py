import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# ======================
# CREATE OUTPUT FOLDER
# ======================
os.makedirs("output", exist_ok=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("data/Mall_Customers.csv")

df.columns = ["CustomerID", "Gender", "Age", "Income", "SpendingScore"]
df.drop("CustomerID", axis=1, inplace=True)

# ======================
# ENCODE DATA
# ======================
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# ======================
# FEATURE ENGINEERING
# ======================
df["SpendingPower"] = df["Income"] * df["SpendingScore"]

features = ["Gender", "Age", "Income", "SpendingScore", "SpendingPower"]
X = df[features]

# ======================
# SCALING
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# KMEANS CLUSTERING
# ======================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["KMeansCluster"] = kmeans.fit_predict(X_scaled)

# ======================
# DBSCAN CLUSTERING (ADVANCED)
# ======================
dbscan = DBSCAN(eps=1.5, min_samples=5)
df["DBSCANCluster"] = dbscan.fit_predict(X_scaled)

# ======================
# PCA VISUALIZATION
# ======================
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

df["PCA1"] = components[:, 0]
df["PCA2"] = components[:, 1]

# ======================
# PLOT CLUSTERS
# ======================
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["KMeansCluster"], palette="Set2")
plt.title("Customer Segmentation (KMeans)")
plt.show()

# ======================
# SAVE OUTPUT
# ======================
df.to_csv("output/segmented_customers.csv", index=False)

print("\nPROJECT RUN SUCCESSFUL 🚀")
print("File saved in: output/segmented_customers.csv")