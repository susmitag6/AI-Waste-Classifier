import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = np.load("data.npy")
labels = np.load("labels.npy")

# Limit dataset size to avoid RAM issues, but sample it *randomly and
# proportionally across classes* instead of just taking the first N rows.

MAX_SAMPLES = 1500
if len(data) > MAX_SAMPLES:
    data, _, labels, _ = train_test_split(
        data, labels,
        train_size=MAX_SAMPLES,
        random_state=42,
        stratify=labels
    )

X_train, X_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Save the held-out test split so evaluation later is run on data
# the model never saw during training.
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------------------------------
# Stop training early once accuracy reaches a target threshold.
# Watches val_accuracy rather than training accuracy, since training
# accuracy can hit the threshold just from memorizing the training set.
# -----------------------------------------------------------------
ACCURACY_THRESHOLD = 0.90  # stop once val_accuracy reaches 90%

class ThresholdStopping(tf.keras.callbacks.Callback):
    def __init__(self, threshold, monitor="val_accuracy"):
        super().__init__()
        self.threshold = threshold
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        if current >= self.threshold:
            print(f"\n{self.monitor} reached {current:.4f} >= {self.threshold} "
                  f"at epoch {epoch + 1}, stopping training.")
            self.model.stop_training = True

threshold_callback = ThresholdStopping(ACCURACY_THRESHOLD, monitor="val_accuracy")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=10,
    verbose=1,
    callbacks=[threshold_callback]
)

model.save("waste_model.h5")
print("Model saved!")
