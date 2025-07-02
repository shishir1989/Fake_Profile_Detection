import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load MNIST dataset
# ----------------------------
def load_data():
    df = pd.read_csv('mnist_train.csv') #we should give the file for dataset.
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:].reshape(-1, 28, 28) / 255.0  # Normalize
    y = data[:, 0]
    return X, y

# ----------------------------
# Convolution Operation
# ----------------------------
def convolve(image, kernel):
    kernel_size = kernel.shape[0]
    result_dim = image.shape[0] - kernel_size + 1
    result = np.zeros((result_dim, result_dim))

    for i in range(result_dim):
        for j in range(result_dim):
            region = image[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.sum(region * kernel)
    return result

# ----------------------------
# ReLU Activation
# ----------------------------
def relu(x):
    return np.maximum(0, x)

# ----------------------------
# Max Pooling
# ----------------------------
def max_pool(feature_map, size=2, stride=2):
    result_dim = (feature_map.shape[0] - size) // stride + 1
    pooled = np.zeros((result_dim, result_dim))
    for i in range(0, feature_map.shape[0] - size + 1, stride):
        for j in range(0, feature_map.shape[1] - size + 1, stride):
            region = feature_map[i:i+size, j:j+size]
            pooled[i//stride, j//stride] = np.max(region)
    return pooled

# ----------------------------
# Fully Connected Layer
# ----------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ----------------------------
# Forward Propagation (Single Sample)
# ----------------------------
def predict(image, kernels, weights):
    # Convolution
    conv_outputs = []
    for kernel in kernels:
        conv = convolve(image, kernel)
        conv = relu(conv)
        pooled = max_pool(conv)
        conv_outputs.append(pooled)
    
    # Flatten
    flat = np.concatenate([f.ravel() for f in conv_outputs])
    
    # Fully connected
    logits = np.dot(weights, flat)
    probs = softmax(logits)
    return probs

# ----------------------------
# Main Code
# ----------------------------
if __name__ == "__main__":
    # Load a small subset of data
    X, y = load_data()
    X, y = X[:100], y[:100]

    # Define random filters (3 kernels)
    kernels = [np.random.randn(3, 3) for _ in range(3)]

    # Flattened size after pooling: (28-2)/2 = 13x13 per kernel
    fc_input_size = 3 * 13 * 13
    weights = np.random.randn(10, fc_input_size)

    # Test on one image
    index = 0
    probs = predict(X[index], kernels, weights)
    pred_label = np.argmax(probs)

    print("Actual Label:", y[index])
    print("Predicted Label:", pred_label)

    plt.imshow(X[index], cmap='gray')
    plt.title(f"Prediction: {pred_label}")
    plt.show()
