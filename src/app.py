import streamlit as st

from models import build_model, evaluate_model, train_model, user_predict
from processor import load_data, preprocess_data
from view import about, get_model_param_from_user, get_pred_inputs_from_user


def app():
    # Set Streamlit page configuration and display app information
    st.set_page_config(
        layout="wide", page_icon=":hospital:", page_title="Heart Attack Risk Classifier"
    )
    st.set_option("deprecation.showPyplotGlobalUse", False)

    about()

    # Load and preprocess data
    data = load_data()
    X, Y = preprocess_data(data)

    # Select a classifier from the sidebar
    classifier_name = st.sidebar.selectbox(
        "Select Classifier: ",
        (
            "SVM",
            "KNN",
            "Logistic Regression",
            "Decision Trees",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
        ),
    )

    # Get model parameters from the user
    params = get_model_param_from_user(classifier_name)
    # Build the classifier model
    clf = build_model(classifier_name, params)
    # Train the model if the "Train the Model" button is clicked
    
    Y_pred, Y_test = train_model(clf, X, Y)
    evaluate_model(Y_pred, Y_test)
 
    # Get user input for prediction
    user_input = get_pred_inputs_from_user(X)
    user_predict(clf, X, user_input)
   

if __name__ == "__main__":
    app()
