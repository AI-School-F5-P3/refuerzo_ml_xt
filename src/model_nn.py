import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

data = pd.read_csv("../data/clean_file.csv")

# X = data.drop(columns=['custcat'])
# y = data['custcat']

X = data[["ed", "tenure", "employ", "reside", "income", "marital", "address"]]
y = data["custcat"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = to_categorical(y - 1, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida en prueba: {loss:.4f}")
print(f"Precisión en prueba: {accuracy:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1) + 1

y_test_classes = np.argmax(y_test, axis=1) + 1
print(classification_report(y_test_classes, y_pred_classes, digits=4))