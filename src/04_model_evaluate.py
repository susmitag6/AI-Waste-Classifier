from pathlib import Path
import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------
# Load data
# -----------------------
data = np.load("data.npy")
labels = np.load("labels.npy")

#Limit dataset for testing (debug mode)
data = data[:1000] #Smaller data sise used to overcome RAM issue
labels = labels[:1000]
# Normalize if your model was trained on normalized images
data = data.astype("float32") / 255.0

# -----------------------
# Load model
# -----------------------
model = tf.keras.models.load_model("waste_model.h5")

# -----------------------
# Evaluate model
# -----------------------
loss, accuracy = model.evaluate(data, labels, verbose=0)

print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy:.4f}")

# -----------------------
# Predictions
# -----------------------
y_pred = model.predict(data, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)


# -----------------------
# Metrics
# -----------------------
precision = precision_score(labels, y_pred_classes, average="weighted")
recall = recall_score(labels, y_pred_classes, average="weighted")
f1 = f1_score(labels, y_pred_classes, average="weighted")

report = classification_report(labels, y_pred_classes)

# -----------------------
# Print Summary
# -----------------------
print("\n================ MODEL EVALUATION ================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Loss     : {loss:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:\n")
print(report)

# -----------------------
# Save Report
# -----------------------
os.makedirs("reports", exist_ok=True)

with open("reports/model_evaluation.txt", "w") as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("========================\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Loss     : {loss:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)
# -----------------------
# Confusion Matrix
# -----------------------
cm = confusion_matrix(labels, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Create figures folder if it doesn't exist
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# Save figure
plt.savefig(figures_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")

plt.close()

print("Confusion matrix saved to figures/confusion_matrix.png")
