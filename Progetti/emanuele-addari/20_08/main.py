from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('20_08/dataset/Mall_Customers.csv')

### KMEANS ###

encoder = LabelEncoder()
df["Genre"] = encoder.fit_transform(df["Genre"])

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])

### Annual income / Spending score ###

### KMeans Clustering ###

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_scaled)

centroidi = kmeans.cluster_centers_
cmap = ListedColormap(colors=['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'], N=kmeans.n_clusters)

plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap=cmap)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
for i, center in enumerate(centroidi):
    orig_centroid = scaler.inverse_transform(center.reshape(1, -1))[0]
    plt.text(orig_centroid[1], orig_centroid[2], str(i))
plt.show()
plt.savefig('20_08/annual_income_cluster_plot.png')

### DBSCAN ###

dbscan = DBSCAN(eps=0.5, min_samples=5)
df["DBSCAN_Cluster"] = dbscan.fit_predict(df_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['DBSCAN_Cluster'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
for i, center in enumerate(centroidi):
    orig_centroid = scaler.inverse_transform(center.reshape(1, -1))[0]
    plt.text(orig_centroid[1], orig_centroid[2], str(i))
plt.show()
plt.savefig('20_08/annual_income_dbscan_cluster_plot.png')

### Calcolo centroidi ###

labels = df["Cluster"]

for i in range(kmeans.n_clusters):
    mask = (labels == i)
    dist_cluster = np.linalg.norm(df_scaled[mask] - centroidi[i], axis=1)
    avg_dist_cluster = dist_cluster.mean()
    print(f"distanza media centroide da cluster {i}: {avg_dist_cluster}")


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['Age'],
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap=cmap
)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['Age'],
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['DBSCAN_Cluster'],
)
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.colorbar(scatter, ax=ax, label='DBSCAN_Cluster')
plt.show()