import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------
# Load and prepare the image
# ------------------------------------------
def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure 3 channels
    img_np = np.array(image)
    return img_np

# ------------------------------------------
# Flatten image pixels to RGB vectors
# ------------------------------------------
def preprocess_image(img_np):
    w, h, c = img_np.shape
    pixels = img_np.reshape((w * h, c))
    return pixels, w, h

# ------------------------------------------
# K-Means Clustering (from scratch)
# ------------------------------------------
def initialize_centroids(pixels, k):
    np.random.seed(42)
    random_idxs = np.random.choice(len(pixels), size=k, replace=False)
    return pixels[random_idxs].astype(float)

def assign_clusters(pixels, centroids):
    distances = np.linalg.norm(pixels[:, None] - centroids, axis=2)  # (n_pixels x k)
    return np.argmin(distances, axis=1)

def update_centroids(pixels, labels, k):
    return np.array([pixels[labels == i].mean(axis=0) if np.any(labels == i) else np.zeros(3) for i in range(k)])

def kmeans(pixels, k, max_iter=100, tol=1e-4):
    centroids = initialize_centroids(pixels, k)
    for _ in range(max_iter):
        labels = assign_clusters(pixels, centroids)
        new_centroids = update_centroids(pixels, labels, k)
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    return centroids, labels

# ------------------------------------------
# Recreate image using cluster centroids
# ------------------------------------------
def recreate_image(centroids, labels, w, h):
    segmented_pixels = centroids[labels].astype(np.uint8)
    segmented_image = segmented_pixels.reshape((w, h, 3))
    return segmented_image

# ------------------------------------------
# Visualization
# ------------------------------------------
def visualize(original, segmented):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented)
    plt.title("Segmented Image (K-Means)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ------------------------------------------
# Main function
# ------------------------------------------
def main():
    image_path = '12.jpg'  # Replace with your image filename
    k = 6  # You can experiment with 3–6 clusters depending on image

    original_img = load_image(image_path)
    pixels, w, h = preprocess_image(original_img)
    centroids, labels = kmeans(pixels, k=k)
    segmented_img = recreate_image(centroids, labels, w, h)

    visualize(original_img, segmented_img)

# Run
if __name__ == '__main__':
    main()
