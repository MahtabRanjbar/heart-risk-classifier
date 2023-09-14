import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error)
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def build_model(classifier_name, params):
    """
    Build and return a machine learning model based on the selected classifier name and parameters.

    Args:
        classifier_name (str): The name of the classifier.
        params (dict): A dictionary containing the parameters for the classifier.

    Returns:
        object: The built machine learning model.

    """
    if classifier_name == "Logistic Regression":
        clf = LogisticRegression(C=params["R"], max_iter=params["MI"])
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == "SVM":
        clf = SVC(kernel=params["kernel"], C=params["C"])
    elif classifier_name == "Decision Trees":
        clf = DecisionTreeClassifier(max_depth=params["M"], criterion=params["C"])
    elif classifier_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["N"], max_depth=params["M"], criterion=params["C"]
        )
    elif classifier_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(
            n_estimators=params["N"],
            learning_rate=params["LR"],
            loss=params["L"],
            max_depth=params["M"],
        )
    elif classifier_name == "XGBoost":
        clf = XGBClassifier(
            booster="gbtree",
            n_estimators=params["N"],
            max_depth=params["M"],
            learning_rate=params["LR"],
            objective=params["O"],
            gamma=params["G"],
            reg_alpha=params["A"],
            reg_lambda=params["L"],
            colsample_bytree=params["CS"],
        )
    return clf


def train_model(clf, X, Y):
    """
    Train the machine learning model using the provided data.

    Args:
        clf (object): The machine learning model.
        X (array-like): The input features.
        Y (array-like): The target variable.

    Returns:
        tuple: A tuple containing the predicted labels and the true labels.

    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=65
    )

    # MinMax Scaling / Normalization of data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    return Y_pred, Y_test

def evaluate_model(Y_pred, Y_test):
    """
    Evaluate the performance of the machine learning model and display the evaluation metrics.

    Args:
        Y_pred (array-like): The predicted labels.
        Y_test (array-like): The true labels.

    """
    c1, c2 = st.columns((4, 3))
    acc = accuracy_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    precision, recall, fscore, _ = score(Y_test, Y_pred, pos_label=1, average="binary")
    c1.subheader("Evaluation Report of Model: ")
    c1.text(
        "Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %\nMean Squared Error: {}".format(
            round(precision, 3),
            round(recall, 3),
            round(fscore, 3),
            round((acc * 100), 3),
            round((mse), 3),
        )
    )

    # c1, c2 = st.columns((4,3))
    # Output plot
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(Y_pred)), Y_pred, color="yellow", lw=5, label="Predictions")
    plt.scatter(range(len(Y_test)), Y_test, color="red", label="Actual")
    plt.title("Prediction Values vs Real Values")
    plt.legend()
    plt.grid(True)
    st.pyplot()

    cm = confusion_matrix(Y_test, Y_pred)
    class_label = ["High-risk", "Low-risk"]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)

    plt.figure(figsize=(12, 7.5))
    sns.heatmap(df_cm, annot=True, cmap="Pastel1", linewidths=2, fmt="d")
    plt.title("Confusion Matrix", fontsize=15)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    c2.pyplot()


def user_predict(_clf, data, user_val):
    """
    Predicts the risk status based on user values using a classifier.

    Parameters:
        clf (classifier object): The classifier model used for prediction.
        data (pandas.DataFrame): The dataset containing the columns used for prediction.
        user_val (dict): A dictionary containing the user's values for each column used in prediction.

    Returns:
        None

    Displays the predicted risk status using the Streamlit library.
    """

    pred = _clf.predict([[user_val[col] for col in data.columns]])

    st.subheader("Your Status: ")
    if pred == 0:
        st.write(pred[0], " - You are not at high risk :)")
    else:
        st.write(pred[0], " - You are at high risk :(")
