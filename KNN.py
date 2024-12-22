from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


# 1. 加载JS散度矩阵
def load_js_divergence_matrix(file_path):
    js_divergence_matrix = pd.read_csv(file_path, header=None)
    return js_divergence_matrix


if __name__ == "__main__":
    # 自定义文件路径
    file_path = "180sample/group_0_0_0_0_0_1.csv"

    # 加载JS散度矩阵
    js_divergence_matrix = load_js_divergence_matrix(file_path)

    # 提取样本名称 (第一行和第一列)
    sample_names = js_divergence_matrix.iloc[1:, 0].values

    # 去掉第一列和第一行，保留纯数值部分
    js_divergence_matrix_numeric = js_divergence_matrix.iloc[1:, 1:].to_numpy()

    # 使用 MinMaxScaler 对矩阵进行归一化
    scaler = MinMaxScaler()
    normalized_js_divergence_matrix = scaler.fit_transform(js_divergence_matrix_numeric)

    # 2. 计算KNN距离矩阵
    knn = NearestNeighbors(n_neighbors=len(normalized_js_divergence_matrix), metric='euclidean')
    knn.fit(normalized_js_divergence_matrix)
    distances, _ = knn.kneighbors(normalized_js_divergence_matrix)
    distance_matrix = np.array(distances)

    # 3. 使用层次聚类算法进行聚类
    best_score = -np.inf
    best_num_clusters = 2
    best_labels = None

    silhouette_scores = []
    calinski_scores = []

    for num_clusters in range(2, 10):  # 遍历所有可选K值
        clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='average')
        labels = clustering.fit_predict(normalized_js_divergence_matrix)

        silhouette_avg = silhouette_score(normalized_js_divergence_matrix, labels)
        calinski_score = calinski_harabasz_score(normalized_js_divergence_matrix, labels)

        silhouette_scores.append(silhouette_avg)
        calinski_scores.append(calinski_score)

    # 4. 归一化评分标准
    silhouette_scores = np.array(silhouette_scores)
    calinski_scores = np.array(calinski_scores)

    silhouette_scores_normalized = (silhouette_scores - silhouette_scores.min()) / (
                silhouette_scores.max() - silhouette_scores.min())
    calinski_scores_normalized = (calinski_scores - calinski_scores.min()) / (
                calinski_scores.max() - calinski_scores.min())

    scores = silhouette_scores_normalized + calinski_scores_normalized

    best_num_clusters = np.argmax(scores) + 2  # 因为我们从2开始计数
    best_labels = AgglomerativeClustering(n_clusters=best_num_clusters, affinity='euclidean',
                                          linkage='average').fit_predict(normalized_js_divergence_matrix)

    # 5. 输出最佳聚类结果
    print(f"Best Number of Clusters: {best_num_clusters}")
    print("Best Cluster Labels:")
    print(best_labels)

    # 计算每个簇中的核心点
    core_points = []
    total_core_count = 0  # 记录所有簇的核心点总数

    for cluster_id in range(best_num_clusters):
        # 获取当前簇的所有点的索引
        cluster_indices = np.where(best_labels == cluster_id)[0]
        cluster_points = normalized_js_divergence_matrix[cluster_indices]

        if len(cluster_points) > 3:
            # 计算簇内距离矩阵
            intra_cluster_distances = pairwise_distances(cluster_points)

            # 计算每个点的平均距离
            avg_distances = np.mean(intra_cluster_distances, axis=1)

            # 核心点: 平均距离最小的点
            core_indices = np.argsort(avg_distances)[:(len(cluster_points) // 2)]  # 选取最中心的一半点
            core_points.extend([sample_names[cluster_indices[i]] for i in core_indices])
            total_core_count += len(core_indices)

        # 如果核心点的数量已经达到所有点的一半，则停止添加
        if total_core_count >= len(normalized_js_divergence_matrix) // 2:
            break

    # 所有未被标记为核心点的点都标记为边缘点
    all_points = set(sample_names)
    core_set = set(core_points)
    boundary_points = list(all_points - core_set)

    # 输出核心点和边缘点
    print("Core Points:")
    for core in core_points:
        print(core)

    print("Boundary Points:")
    for boundary in boundary_points:
        print(boundary)

    # 保存核心点和边缘点到文件
    with open("group_0_0_0_0_0_1_core_points.csv", "w") as core_file:
        for core in core_points:
            core_file.write(f"{core}\n")

    with open("group_0_0_0_0_0_1_boundary_points.csv", "w") as boundary_file:
        for boundary in boundary_points:
            boundary_file.write(f"{boundary}\n")

    # 7. 可视化
    plt.figure(figsize=(10, 8))  # 设置图像尺寸
    plt.scatter(normalized_js_divergence_matrix[:, 0], normalized_js_divergence_matrix[:, 1], c=best_labels,
                cmap='viridis', s=50)  # 增加点的大小
    plt.title('Cluster Result')
    plt.xticks([])  # 去掉 x 轴数字
    plt.yticks([])  # 去掉 y 轴数字
    # plt.xlabel('Feature 1')  # 注释掉 x 轴标签
    # plt.ylabel('Feature 2')  # 注释掉 y 轴标签
    # plt.colorbar()  # 注释掉颜色条
    # plt.savefig('clustering_visualization.png', dpi=1000)  # 保存图像时指定分辨率
    plt.show()

    # ... 省略中间代码 ...

    # 保存最佳聚类结果到文件
    np.savetxt("Best_group_0_0_0_0_0_1_KNN_clusters.csv", best_labels, delimiter=",")

    # 如果需要保存为向量图形
    plt.savefig('clustering_visualization.svg', format='svg')


