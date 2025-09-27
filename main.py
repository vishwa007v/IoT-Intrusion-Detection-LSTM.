import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATA_PATH = "./nbaiot_dataset/"
DEV_LIST = []       # fill with device folders from dataset
TIME_STEP = 10

def load_data(path, devices):
    files = []
    for d in devices:
        d_path = os.path.join(path, d)
        if os.path.isdir(d_path):
            for f in os.listdir(d_path):
                if f.endswith(".csv"):
                    files.append(os.path.join(d_path, f))

    if not files:
        x = np.random.rand(100, 115)
        df = pd.DataFrame(x, columns=[f"f{i}" for i in range(115)])
        df["label"] = np.random.choice(["benign", "attack"], 100)
        return df

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    label_cols = df.columns[-11:]
    df["label"] = df[label_cols].idxmax(axis=1)
    return df.drop(columns=label_cols)

def preprocess(df):
    X = df.drop("label", axis=1).values
    y = df["label"].values
    X = MinMaxScaler().fit_transform(X)
    enc = LabelEncoder()
    y = enc.fit_transform(y)
    y = to_categorical(y)
    return X, y, len(enc.classes_), enc

def make_seq(X, y, steps):
    xs, ys = [], []
    for i in range(len(X) - steps):
        xs.append(X[i:i+steps])
        ys.append(y[i+steps])
    return np.array(xs), np.array(ys)

def build_model(shape, classes):
    m = Sequential()
    m.add(Conv1D(64, 3, activation="relu", input_shape=shape))
    m.add(MaxPooling1D(2))
    m.add(LSTM(100))
    m.add(Dropout(0.5))
    m.add(Dense(classes, activation="softmax"))
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

if __name__ == "__main__":
    data = load_data(DATA_PATH, DEV_LIST)
    X, y, n_classes, encoder = preprocess(data)
    X_seq, y_seq = make_seq(X, y, TIME_STEP)

    if len(X_seq) == 0:
        print("Not enough data.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=np.argmax(y_seq, 1)
    )

    ns, nx, ny = X_train.shape
    smote = SMOTE(random_state=42)
    X2d, y2d = smote.fit_resample(X_train.reshape(ns, nx*ny), np.argmax(y_train, 1))
    X_train = X2d.reshape(-1, nx, ny)
    y_train = to_categorical(y2d, num_classes=n_classes)

    model = build_model((nx, ny), n_classes)
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Acc: {acc*100:.2f}% | Loss: {loss:.4f}")

    preds = np.argmax(model.predict(X_test), 1)
    truth = np.argmax(y_test, 1)
    print(classification_report(truth, preds, target_names=encoder.classes_))
    print(confusion_matrix(truth, preds))
