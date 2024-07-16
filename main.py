import os
import gzip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Function to load images
def load_images(file_path):
    with gzip.open(file_path, 'r') as f:
        f.read(16)  # Skip the header
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        images = data.reshape(-1, 28, 28)
    return images


# Function to load labels
def load_labels(file_path):
    with gzip.open(file_path, 'r') as f:
        f.read(8)  # Skip the header
        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


# Load the dataset
def load_fashion_mnist():
    base_url = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    data_dir = './fashion_mnist_data/'

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download files if not already downloaded
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f'Downloading {file_name}...')
            url = f'{base_url}/{file_name}'
            tf.keras.utils.get_file(file_path, url)

    # Load training and testing data
    train_images = load_images(os.path.join(data_dir, file_names[0]))
    train_labels = load_labels(os.path.join(data_dir, file_names[1]))
    test_images = load_images(os.path.join(data_dir, file_names[2]))
    test_labels = load_labels(os.path.join(data_dir, file_names[3]))

    # Normalize images to the range [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape data to fit the model
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    return (train_images, train_labels), (test_images, test_labels)


# Define the model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Train the model
def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10):
    model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))


# Save the model
def save_model(model, model_dir='saved_model'):
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'fashion_mnist_model.h5'))
    print(f"Model saved successfully at {model_dir}")


# Load the model
def load_saved_model(model_dir='saved_model'):
    model_path = os.path.join(model_dir, 'fashion_mnist_model.h5')
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    else:
        print(f"No model found at {model_path}. Proceeding with model training.")
        return None


# Evaluate the model
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")


# Visualize predictions
def visualize_predictions(model, test_images, test_labels, num_images=5):
    predictions = model.predict(test_images[:num_images])

    for i in range(num_images):
        plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {test_labels[i]}")
        plt.show()


# Main function to run the entire process
def main():
    # Load Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = load_fashion_mnist()

    # Load or build the model
    model = load_saved_model()
    if model is None:
        model = build_model()
        # Train the model if not loaded from disk
        train_model(model, train_images, train_labels, test_images, test_labels)
        # Save the trained model
        save_model(model)
    else:
        # Evaluate the loaded model
        evaluate_model(model, test_images, test_labels)

    # Visualize predictions
    visualize_predictions(model, test_images, test_labels)


if __name__ == '__main__':
    main()
