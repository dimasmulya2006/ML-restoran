# =========================
# IMPORT LIBRARY
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# =========================
# LOAD DATASET (LOCAL)
# =========================
DATA_PATH = "datasetDIYKOMBERv2.0.csv"

df = pd.read_csv(DATA_PATH)

print("\n=== DATASET PREVIEW ===")
print(df.head())


# =========================
# PREPROCESSING
# =========================
# fitur (X) dan label (y)
X = df.drop(columns="packet")
y = df["packet"]

print("\n=== SHAPE DATA ===")
print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n=== DATA SPLIT ===")
print(f"Train: {X_train.shape}")
print(f"Test : {X_test.shape}")


# =========================
# MODEL TRAINING
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# PREDIKSI
# =========================
y_pred = model.predict(X_test)


# =========================
# EVALUASI MODEL
# =========================
print("\n=== EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# =========================
# SAVE MODEL
# =========================
MODEL_PATH = "RandomForest_Model_DYKOMBER.joblib"
joblib.dump(model, MODEL_PATH)

print(f"\nModel tersimpan sebagai: {MODEL_PATH}")


# =========================
# TESTING MANUAL INPUT
# =========================
print("\n=== INPUT MANUAL TEST ===")

try:
    appetizer = int(input("Masukkan kode appetizer: "))
    main_course = int(input("Masukkan kode main course: "))
    dessert = int(input("Masukkan kode dessert: "))
    drink = int(input("Masukkan kode drink: "))

    input_baru = pd.DataFrame([{
        "appetizer": appetizer,
        "main_course": main_course,
        "dessert": dessert,
        "drink": drink
    }])

    hasil = model.predict(input_baru)

    print("\n=== HASIL REKOMENDASI ===")
    print("Packet yang direkomendasikan:", hasil[0])

except Exception as e:
    print("Terjadi error input:", e)python 