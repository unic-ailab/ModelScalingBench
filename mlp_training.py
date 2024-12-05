import tensorflow as tf
from tensorflow.keras import layers, models
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import csv
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load California Housing dataset (Regression)
print("Loading California Housing dataset...")
X_reg, y_reg = fetch_california_housing(return_X_y=True)
X_reg = X_reg.astype('float32')
y_reg = y_reg.astype('float32')

# Split the regression dataset
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

del X_reg
del y_reg

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784')
X_mnist, y_mnist = mnist.data.astype('float32') / 255.0, mnist.target.astype('int')

X_mnist = np.array(X_mnist)
y_mnist = np.array(y_mnist)

# Split the MNIST dataset
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)

del X_mnist
del y_mnist

# Standardize the regression data
scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# Standardize the MNIST data
scaler_mnist = StandardScaler()
X_train_mnist = scaler_mnist.fit_transform(X_train_mnist.reshape(-1, 28 * 28))
X_test_mnist = scaler_mnist.transform(X_test_mnist.reshape(-1, 28 * 28))

# Create a directory to save the models and weights
model_save_dir = "/content/drive/MyDrive/model_hot_swapping/saved_models"
os.makedirs(model_save_dir, exist_ok=True)

num_written_rows = 0

# Create and open a CSV file to store results
csv_file_path = "<csv_path>"
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Dataset Type', 'Hidden Layers', 'Neurons per Hidden Layer', 'Batch Size', 'Epochs', 'Test Accuracy / MSE Score', 'Inference Time (s)', 'Test Loss', 'Model Name'])
else:
    with open(csv_file_path, 'r') as csv_file:
        # Determine the number of rows already written
        num_written_rows = sum(1 for _ in csv_file) - 1  # Subtract header row

iteration = 1
all_iters = 2 * 10 * 3 * 4 * 3

# Iterate through different dataset types
for dataset_type, X_train, X_test, y_train, y_test in [('Regression', X_train_reg, X_test_reg, y_train_reg, y_test_reg),
                                                       ('MNIST', X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist)]:
    # Iterate through different numbers of layers
    for num_layers in range(1, 11):
        # Iterate through different numbers of neurons per hidden layer
        for neurons_per_layer in [32, 64, 128]:
            # Iterate through different batch sizes
            for batch_size in [32, 64, 128, 256]:
                # Iterate through different epoch sizes
                for epochs in [1, 3, 5]:
                    print(f"{iteration} / {all_iters}")
                    iteration += 1
                    if num_written_rows > 0:
                        # Skip configurations already written to CSV
                        num_written_rows -= 1
                        continue
                    print(f"\nTraining MLP on {dataset_type} with {num_layers} layers, {neurons_per_layer} neurons/layer, batch size {batch_size}, {epochs} epochs.")

                    # Build the MLP model
                    model = models.Sequential()
                    model.add(layers.Flatten(input_shape=(X_train.shape[1],)))  # Input layer

                    for _ in range(num_layers):
                        model.add(layers.Dense(neurons_per_layer, activation='leaky_relu'))  # Hidden layers

                    # Determine the number of classes based on the dataset type
                    if dataset_type == 'Regression':
                        num_classes = 1  # For regression
                        output_activation = 'linear'
                        loss_function = 'mean_squared_error'
                        optimizer = tf.keras.optimizers.Adam()
                        metric = 'mse'
                    else:
                        num_classes = 10  # For multi-class classification (MNIST)
                        output_activation = 'softmax'
                        loss_function = 'sparse_categorical_crossentropy'
                        optimizer = tf.keras.optimizers.Adam()
                        metric = 'accuracy'

                    # Output layer
                    model.add(layers.Dense(num_classes, activation=output_activation))

                    model.compile(optimizer=optimizer,
                                  loss=loss_function,
                                  metrics=[metric])

                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                    # Evaluate the model
                    start_time = time.time()
                    test_loss, test_accuracy_mse = model.evaluate(X_test, y_test, verbose=0)
                    inference_time = time.time() - start_time

                    # Generate a unique name for the model based on configuration
                    model_name = f"model_{dataset_type.lower()}_layers{num_layers}_neurons{neurons_per_layer}_batch{batch_size}_epochs{epochs}.h5"
                    model_path = os.path.join(model_save_dir, model_name)

                    # Save both architecture and weights
                    model.save(model_path)

                    # Write results, configurations, and model name to CSV file
                    with open(csv_file_path, 'a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow([dataset_type, num_layers, neurons_per_layer, batch_size, epochs, test_accuracy_mse, inference_time, test_loss, model_name])

                    print(f"Results and models saved to {csv_file_path} and {model_save_dir}")