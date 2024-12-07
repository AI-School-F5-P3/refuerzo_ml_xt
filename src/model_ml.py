import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv("../data/clean_file.csv")

def split_data_7f(data: pd.DataFrame):
    X = data[["ed", "tenure", "employ", "reside", "income", "marital", "address"]]
    y = data["custcat"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_pipeline():
    pipeline = Pipeline([
        ('preprocessor', StandardScaler()),
        ('classifier', SVC(C=1, 
                           class_weight='balanced', 
                           kernel='linear', 
                           gamma='scale', 
                           random_state=42))
        ])
    return pipeline

def save_model(model, filename):
    joblib.dump(model, filename)
    print("Modelo guardado: ", filename)

def train_model():
    X_train, X_test, y_train, y_test = split_data_7f(data)
    model = create_pipeline()
    model.fit(X_train, y_train)
    #predict
    y_pred = model.predict(X_test)
    #metrics
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", test_accuracy)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Overfitting:", train_accuracy - test_accuracy)
    save_model(model, "../models/model_svc_7f.joblib")

if __name__ == '__main__':
    train_model()

