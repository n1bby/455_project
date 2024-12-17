import tensorflow as tf
import dataset
import numpy as np
import random
import sys

# Constants
NUM_CLASSES = len(dataset.CLASSES)
IMAGE_SIZE = 100

# Placeholder-like Input
def get_dataset_inputs(batch_size):
    images_placeholder = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="images")
    labels_placeholder = tf.keras.Input(shape=(), dtype=tf.int32, name="labels")
    return images_placeholder, labels_placeholder

# Weight and Bias Initializers
def get_weights(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name="Weights")

def get_biases(shape):
    return tf.Variable(tf.zeros(shape), name="Biases")

# Convolutional Layer
def conv_layer(inputs, filter_size, name, strides=(1, 1), pool_size=2, padding="SAME"):
    with tf.name_scope(name):
        conv = tf.keras.layers.Conv2D(
            filters=filter_size[-1],
            kernel_size=filter_size[:2],
            strides=strides,
            padding=padding,
            activation=tf.nn.leaky_relu
        )(inputs)
        pool = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_size, pool_size))(conv)
        return pool

# Dense Layer
def dense_layer(inputs, units, name):
    return tf.keras.layers.Dense(units, activation=tf.nn.leaky_relu, name=name)(inputs)

# Build the Model
def build_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="images")

    x = conv_layer(inputs, [20, 20, 3, 10], "Conv1", pool_size=4)
    x = conv_layer(x, [5, 5, 10, 20], "Conv2")
    x = tf.keras.layers.Flatten()(x)
    x = dense_layer(x, 1000, "Dense1")
    x = dense_layer(x, 100, "Dense2")
    logits = tf.keras.layers.Dense(NUM_CLASSES, activation=None, name="Softmax_Linear")(x)

    return tf.keras.Model(inputs=inputs, outputs=logits)

# Define Loss and Accuracy
def define_loss(logits, labels):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return loss_fn(labels, logits)

# Evaluation
def evaluate_model(model, dataset, batch_size):
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for images, labels in dataset.batch(batch_size):
        predictions = model(images, training=False)
        accuracy.update_state(labels, predictions)
    print(f"Validation Accuracy: {accuracy.result().numpy():.4f}")
    return accuracy.result()

# Training Loop
def run_training(batch_size, learning_rate, epochs):
    # Prepare the dataset
    train_images, train_labels = dataset.get_training_data()
    val_images, val_labels = dataset.get_validation_data(batch_size)
    test_images, test_labels = dataset.get_test_data(batch_size)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    # Build and compile the model
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Training process
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_loss = tf.keras.metrics.Mean()
        for step, (images, labels) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss_value = loss_fn(labels, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.update_state(loss_value)

        print(f"Loss: {epoch_loss.result().numpy():.4f}")
        evaluate_model(model, val_ds, batch_size)

    # Evaluate on test set
    print("\nFinal Evaluation on Test Set:")
    evaluate_model(model, test_ds, batch_size)

    # Save the model
    model.save("model_tf2.h5")
    print("Model saved to model_tf2.h5")

# Run Training
run_training(batch_size=89, learning_rate=0.001, epochs=10)

