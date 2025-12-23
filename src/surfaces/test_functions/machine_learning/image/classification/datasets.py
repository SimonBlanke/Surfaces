# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Image classification datasets."""

import numpy as np

# Lazy-loaded datasets
_mnist_data = None
_fashion_mnist_data = None


def _generate_mnist_like():
    """Generate MNIST-like synthetic data for testing without external downloads."""
    np.random.seed(42)
    n_samples = 1000
    n_classes = 10
    img_size = 28

    X = []
    y = []

    for class_idx in range(n_classes):
        for _ in range(n_samples // n_classes):
            # Create distinct patterns for each digit
            img = np.zeros((img_size, img_size))

            if class_idx == 0:  # Circle
                center = img_size // 2
                for i in range(img_size):
                    for j in range(img_size):
                        if 8 < np.sqrt((i - center) ** 2 + (j - center) ** 2) < 12:
                            img[i, j] = 1.0
            elif class_idx == 1:  # Vertical line
                img[:, img_size // 2 - 1 : img_size // 2 + 2] = 1.0
            elif class_idx == 2:  # Horizontal line at top and diagonal
                img[5:8, 5:23] = 1.0
                for i in range(15):
                    img[8 + i, 5 + i] = 1.0
            elif class_idx == 3:  # Two horizontal arcs
                img[5:8, 8:20] = 1.0
                img[12:15, 8:20] = 1.0
                img[20:23, 8:20] = 1.0
            elif class_idx == 4:  # L-shape and vertical
                img[5:20, 5:8] = 1.0
                img[12:15, 5:20] = 1.0
                img[5:23, 18:21] = 1.0
            elif class_idx == 5:  # S-shape
                img[5:8, 8:20] = 1.0
                img[8:14, 5:8] = 1.0
                img[12:15, 8:20] = 1.0
                img[14:21, 18:21] = 1.0
                img[19:22, 8:20] = 1.0
            elif class_idx == 6:  # 6-shape
                img[5:22, 5:8] = 1.0
                img[5:8, 5:20] = 1.0
                img[12:15, 5:20] = 1.0
                img[19:22, 5:20] = 1.0
                img[12:22, 18:21] = 1.0
            elif class_idx == 7:  # 7-shape
                img[5:8, 5:23] = 1.0
                for i in range(18):
                    if 5 + i < img_size and 20 - i // 2 >= 0:
                        img[7 + i, max(10, 20 - i // 2) : 23] = 0.0
                        img[7 + i, max(5, 18 - i // 2) : min(img_size, 21 - i // 2)] = 1.0
            elif class_idx == 8:  # 8-shape (two circles)
                center = img_size // 2
                for i in range(img_size):
                    for j in range(img_size):
                        d1 = np.sqrt((i - 9) ** 2 + (j - center) ** 2)
                        d2 = np.sqrt((i - 19) ** 2 + (j - center) ** 2)
                        if 4 < d1 < 7 or 4 < d2 < 7:
                            img[i, j] = 1.0
            else:  # 9-shape
                img[5:15, 18:21] = 1.0
                img[5:8, 8:20] = 1.0
                img[12:15, 8:20] = 1.0
                img[5:15, 5:8] = 1.0
                img[15:23, 18:21] = 1.0

            # Add noise
            img += 0.1 * np.random.randn(img_size, img_size)
            img = np.clip(img, 0, 1)

            X.append(img.flatten())
            y.append(class_idx)

    X = np.array(X)
    y = np.array(y)

    # Shuffle
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]


def _generate_fashion_like():
    """Generate Fashion-MNIST-like synthetic data."""
    np.random.seed(43)
    n_samples = 1000
    n_classes = 10
    img_size = 28

    X = []
    y = []

    # Fashion items: T-shirt, Trouser, Pullover, Dress, Coat,
    # Sandal, Shirt, Sneaker, Bag, Ankle boot
    for class_idx in range(n_classes):
        for _ in range(n_samples // n_classes):
            img = np.zeros((img_size, img_size))

            if class_idx == 0:  # T-shirt (wide top, narrow bottom)
                img[4:8, 4:24] = 1.0  # shoulders
                img[8:20, 8:20] = 1.0  # body
            elif class_idx == 1:  # Trouser (two legs)
                img[4:24, 6:12] = 1.0
                img[4:24, 16:22] = 1.0
                img[4:8, 6:22] = 1.0
            elif class_idx == 2:  # Pullover (similar to T-shirt but longer)
                img[4:8, 2:26] = 1.0
                img[8:22, 8:20] = 1.0
                img[4:14, 2:8] = 1.0
                img[4:14, 20:26] = 1.0
            elif class_idx == 3:  # Dress (triangular)
                for i in range(20):
                    width = 4 + i // 2
                    img[4 + i, 14 - width : 14 + width] = 1.0
            elif class_idx == 4:  # Coat (wide rectangle)
                img[4:24, 4:24] = 1.0
                img[4:16, 2:6] = 1.0
                img[4:16, 22:26] = 1.0
            elif class_idx == 5:  # Sandal (thin straps)
                img[18:24, 6:22] = 0.5
                img[10:12, 6:22] = 1.0
                img[10:24, 8:10] = 1.0
                img[10:24, 18:20] = 1.0
            elif class_idx == 6:  # Shirt (like T-shirt with collar)
                img[4:8, 6:22] = 1.0
                img[8:22, 8:20] = 1.0
                img[4:10, 12:16] = 0.5  # collar
            elif class_idx == 7:  # Sneaker (shoe shape)
                img[16:24, 4:24] = 1.0
                img[12:18, 4:10] = 1.0
            elif class_idx == 8:  # Bag (rectangle with handle)
                img[8:22, 6:22] = 1.0
                img[4:10, 10:12] = 1.0
                img[4:10, 16:18] = 1.0
            else:  # Ankle boot (tall shoe)
                img[12:24, 6:22] = 1.0
                img[4:14, 6:14] = 1.0

            # Add noise and variation
            img += 0.15 * np.random.randn(img_size, img_size)
            img = np.clip(img, 0, 1)

            X.append(img.flatten())
            y.append(class_idx)

    X = np.array(X)
    y = np.array(y)

    indices = np.random.permutation(len(y))
    return X[indices], y[indices]


def mnist_data():
    """MNIST-like handwritten digits dataset (1000 samples, 784 features, 10 classes).

    Synthetic data mimicking MNIST structure for testing without external downloads.

    Returns
    -------
    tuple
        (X, y) where X is flattened images (n_samples, 784) and y is digit labels (0-9).
    """
    global _mnist_data
    if _mnist_data is None:
        _mnist_data = _generate_mnist_like()
    return _mnist_data


def fashion_mnist_data():
    """Fashion-MNIST-like dataset (1000 samples, 784 features, 10 classes).

    Synthetic data mimicking Fashion-MNIST structure for testing without external downloads.

    Returns
    -------
    tuple
        (X, y) where X is flattened images (n_samples, 784) and y is fashion item labels.
    """
    global _fashion_mnist_data
    if _fashion_mnist_data is None:
        _fashion_mnist_data = _generate_fashion_like()
    return _fashion_mnist_data


# Registry for easy access
DATASETS = {
    "mnist": mnist_data,
    "fashion_mnist": fashion_mnist_data,
}
