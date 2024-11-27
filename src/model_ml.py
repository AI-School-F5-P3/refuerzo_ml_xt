import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("../data/clean_file.csv")

def split_data(data: pd.DataFrame):

    X = data.drop("custcat", axis=1)
    y = data["custcat"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def processing_data(X_train, X_test):
    # Estandarizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

def get_models():
    models = {
        "SVC": svm.SVC(**{'C': 0.5, 'degree': 1, 'gamma': 0.01, 'kernel': 'poly'}),
        "random_foresst": RandomForestClassifier()
    }

def train_model():
    X_train, X_test, y_train, y_test = split_data(data)
    

