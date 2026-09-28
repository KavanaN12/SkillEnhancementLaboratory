import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==================================================
# 1. Reproducibility
# ==================================================

np.random.seed(42)
tf.random.set_seed(42)


# ==================================================
# 2. File paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "student_performance.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ==================================================
# 3. Load dataset
# ==================================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        "student_performance.csv was not found.\n"
        "Run generate_dataset.py first."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("ACADEMIC PERFORMANCE PREDICTION USING DNN")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five records:")
print(df.head())


# ==================================================
# 4. Select features and target
# ==================================================
#
# IMPORTANT:
# Only the two features specified in the assignment
# are used.
#
# Study Hours
# Attendance Percentage
#
# Target:
# Pass / Fail
# ==================================================

X = df[
    [
        "study_hours",
        "attendance"
    ]
]

y = df["result"]


# ==================================================
# 5. Train / Validation / Test split
# ==================================================
#
# 80% Training
# 10% Validation
# 10% Testing
# ==================================================

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    random_state=42,

    stratify=y_temp
)


print("\nData split:")
print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Testing samples    : {len(X_test)}")


# ==================================================
# 6. Feature scaling
# ==================================================
#
# StandardScaler transforms the features so that
# they have a comparable scale.
#
# IMPORTANT:
# The scaler is fitted ONLY on training data.
# This prevents data leakage.
# ==================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ==================================================
# 7. Build TensorFlow DNN
# ==================================================

model = tf.keras.Sequential([

    # Input layer
    tf.keras.layers.Input(
        shape=(2,)
    ),

    # Hidden layer 1
    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    # Hidden layer 2
    tf.keras.layers.Dense(
        8,
        activation="relu"
    ),

    # Output layer
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==================================================
# 8. Compile model
# ==================================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


print("\nModel architecture:")
model.summary()


# ==================================================
# 9. Early stopping
# ==================================================
#
# Training stops if validation loss stops improving.
# ==================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True
)


# ==================================================
# 10. Train the DNN
# ==================================================

print("\nStarting training...\n")

history = model.fit(

    X_train_scaled,

    y_train,

    validation_data=(
        X_val_scaled,
        y_val
    ),

    epochs=100,

    batch_size=32,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ==================================================
# 11. Evaluate on test data
# ==================================================

test_loss, test_accuracy = model.evaluate(

    X_test_scaled,

    y_test,

    verbose=0
)


# ==================================================
# 12. Generate predictions
# ==================================================

probabilities = model.predict(

    X_test_scaled,

    verbose=0
).ravel()


# Convert probability to class
#
# >= 0.50 → Pass
# <  0.50 → Fail

predictions = (
    probabilities >= 0.50
).astype(int)


# ==================================================
# 13. Print model results
# ==================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n")
print("=" * 60)
print("FINAL MODEL RESULTS")
print("=" * 60)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy:.4f}"
    f" ({accuracy * 100:.2f}%)"
)


# ==================================================
# 14. Classification report
# ==================================================

print("\nClassification Report:")

print(
    classification_report(

        y_test,

        predictions,

        target_names=[
            "Fail",
            "Pass"
        ],

        digits=4
    )
)


# ==================================================
# 15. Confusion matrix
# ==================================================

cm = confusion_matrix(

    y_test,

    predictions
)


print("Confusion Matrix:")

print(cm)


# ==================================================
# 16. Save trained model
# ==================================================

model_path = os.path.join(

    RESULTS_DIR,

    "academic_performance_dnn.keras"
)

model.save(
    model_path
)


# ==================================================
# 17. Save scaler information
# ==================================================
#
# predict.py uses these values so that new student
# data is scaled exactly like the training data.
# ==================================================

np.save(

    os.path.join(
        RESULTS_DIR,
        "scaler_mean.npy"
    ),

    scaler.mean_
)


np.save(

    os.path.join(
        RESULTS_DIR,
        "scaler_scale.npy"
    ),

    scaler.scale_
)


# ==================================================
# 18. Training Accuracy graph
# ==================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"
)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "accuracy.png"
    ),

    dpi=150
)

plt.close()


# ==================================================
# 19. Training Loss graph
# ==================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(

    history.history["loss"],

    label="Training Loss"
)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "loss.png"
    ),

    dpi=150
)

plt.close()


# ==================================================
# 20. Confusion Matrix graph
# ==================================================

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Fail",
        "Pass"
    ]
)

disp.plot()

plt.title(
    "Confusion Matrix"
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),

    dpi=150
)

plt.close()


# ==================================================
# 21. Save test predictions
# ==================================================

test_results = X_test.copy()

test_results["actual_result"] = (
    y_test.values
)

test_results["predicted_probability"] = (
    probabilities
)

test_results["predicted_result"] = (
    predictions
)


# Convert numerical labels into readable labels

test_results["actual_result"] = (
    test_results["actual_result"]
    .map({
        0: "Fail",
        1: "Pass"
    })
)


test_results["predicted_result"] = (
    test_results["predicted_result"]
    .map({
        0: "Fail",
        1: "Pass"
    })
)


test_results.to_csv(

    os.path.join(
        RESULTS_DIR,
        "test_predictions.csv"
    ),

    index=False
)


# ==================================================
# 22. Final message
# ==================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nGenerated files:")

print(
    "✓ results/academic_performance_dnn.keras"
)

print(
    "✓ results/scaler_mean.npy"
)

print(
    "✓ results/scaler_scale.npy"
)

print(
    "✓ results/accuracy.png"
)

print(
    "✓ results/loss.png"
)

print(
    "✓ results/confusion_matrix.png"
)

print(
    "✓ results/test_predictions.csv"
)