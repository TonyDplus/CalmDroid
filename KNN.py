from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances



def load_js_divergence_matrix(file_path):
    js_divergence_matrix = pd.read_csv(file_path, header=None)
    return js_divergence_matrix


if __name__ == "__main__":
    
    file_path = "180sample/group_0_0_0_0_0_1.csv"

 
    js_divergence_matrix = load_js_divergence_matrix(file_path)

   
    sample_names = js_divergence_matrix.iloc[1:, 0].values

   
    js_divergence_matrix_numeric = js_divergence_matrix.iloc[1:, 1:].to_numpy()

   
    scaler = MinMaxScaler()
    normalized_js_divergence_matrix = scaler.fit_transform(js_divergence_matrix_numeric)

   
    knn = NearestNeighbors(n_neighbors=len(normalized_js_divergence_matrix), metric='euclidean')
    knn.fit(normalized_js_divergence_matrix)
    distances, _ = knn.kneighbors(normalized_js_divergence_matrix)
    distance_matrix = np.array(distances)

    
    best_score = -np.inf
    best_num_clusters = 2
    best_labels = None

    silhouette_scores = []
    calinski_scores = []

    for num_clusters in range(2, 10): 
        clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='average')
        labels = clustering.fit_predict(normalized_js_divergence_matrix)

        silhouette_avg = silhouette_score(normalized_js_divergence_matrix, labels)
        calinski_score = calinski_harabasz_score(normalized_js_divergence_matrix, labels)

        silhouette_scores.append(silhouette_avg)
        calinski_scores.append(calinski_score)

   
    silhouette_scores = np.array(silhouette_scores)
    calinski_scores = np.array(calinski_scores)

    silhouette_scores_normalized = (silhouette_scores - silhouette_scores.min()) / (
                silhouette_scores.max() - silhouette_scores.min())
    calinski_scores_normalized = (calinski_scores - calinski_scores.min()) / (
                calinski_scores.max() - calinski_scores.min())

    scores = silhouette_scores_normalized + calinski_scores_normalized

    best_num_clusters = np.argmax(scores) + 2  
    best_labels = AgglomerativeClustering(n_clusters=best_num_clusters, affinity='euclidean',
                                          linkage='average').fit_predict(normalized_js_divergence_matrix)

   
    print(f"Best Number of Clusters: {best_num_clusters}")
    print("Best Cluster Labels:")
    print(best_labels)

    
    core_points = []
    total_core_count = 0  

    for cluster_id in range(best_num_clusters):
        
        cluster_indices = np.where(best_labels == cluster_id)[0]
        cluster_points = normalized_js_divergence_matrix[cluster_indices]

        if len(cluster_points) > 3:
           
            intra_cluster_distances = pairwise_distances(cluster_points)

            
            avg_distances = np.mean(intra_cluster_distances, axis=1)

           
            core_indices = np.argsort(avg_distances)[:(len(cluster_points) // 2)]  # 选取最中心的一半点
            core_points.extend([sample_names[cluster_indices[i]] for i in core_indices])
            total_core_count += len(core_indices)

       
        if total_core_count >= len(normalized_js_divergence_matrix) // 2:
            break

   
    all_points = set(sample_names)
    core_set = set(core_points)
    boundary_points = list(all_points - core_set)

   
    print("Core Points:")
    for core in core_points:
        print(core)

    print("Boundary Points:")
    for boundary in boundary_points:
        print(boundary)

    
    with open("group_0_0_0_0_0_1_core_points.csv", "w") as core_file:
        for core in core_points:
            core_file.write(f"{core}\n")

    with open("group_0_0_0_0_0_1_boundary_points.csv", "w") as boundary_file:
        for boundary in boundary_points:
            boundary_file.write(f"{boundary}\n")

   

    np.savetxt("Best_group_0_0_0_0_0_1_KNN_clusters.csv", best_labels, delimiter=",")

   
   


