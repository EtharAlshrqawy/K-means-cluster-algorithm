import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random

class KMeansClustering:
    def __init__(self, k=5, max_iterations=100, tol=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.k, replace=False)
        return X[indices]

    def assign_clusters(self, X, centroids):
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.sum((X - centroids[i])**2, axis=1)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iterations):
            self.labels = self.assign_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, self.labels)
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
        return self

    def predict(self, X):
        return self.assign_clusters(X, self.centroids)

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    height, width, channels = img_array.shape
    pixels = img_array.reshape(-1, channels)
    return img, img_array, pixels

def extract_number_from_clusters(img_array, labels, k):
    height, width, _ = img_array.shape
    cluster_masks = []
    for i in range(k):
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[labels.reshape(height, width) == i] = 255
        cluster_masks.append(mask)
    return cluster_masks

def display_results(original_img, cluster_masks, k):
    original_array = np.array(original_img)
    n_rows = 2 
    n_cols = max(k // 2 + 1, 2)
    plt.figure(figsize=(15, 10))

    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_array)
    plt.title('Original Image')
    plt.axis('off')

    for i, mask in enumerate(cluster_masks):
        plt.subplot(n_rows, n_cols, i + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Cluster {i}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def highlight_digit_cluster(img_array, labels, centroids, target_cluster_idx):
    height, width, _ = img_array.shape
    mask = (labels.reshape(height, width) == target_cluster_idx).astype(np.uint8)
    result_img = np.zeros((height, width, 3), dtype=np.uint8)
    result_img[mask == 1] = [0, 255, 0]
    return result_img

def main():
    image_path = "6.jpg"
    k = 5

    try:
        original_img, img_array, pixels = load_and_preprocess_image(image_path)

        kmeans = KMeansClustering(k=k)
        kmeans.fit(pixels)
        labels = kmeans.labels

        cluster_masks = extract_number_from_clusters(img_array, labels, k)
        display_results(original_img, cluster_masks, k)

        print("Centroids (RGB):")
        for idx, centroid in enumerate(kmeans.centroids):
            print(f"Cluster {idx}: {centroid.astype(int)}")

        digit_cluster_index = 3

        highlighted = highlight_digit_cluster(img_array, labels, kmeans.centroids, digit_cluster_index)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
