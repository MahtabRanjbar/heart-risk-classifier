import streamlit as st


def about():
    # Create a two-column layout
    logo_col, text_col = st.columns([1, 9])

    # Display the logo
    logo_col.image("images/healthcare2.png", width=120)

    # Display the text
    text_col.markdown("<h1 >Heart Attack Risk Classifier </h1>", unsafe_allow_html=True)
    st.markdown(
        """
        Welcome to the Heart Attack Risk Classifier app! This app utilizes machine learning to predict an individual's risk of heart attack based on specific factors. Using a trained classifier model, the app categorizes individuals into low risk or high risk groups."""
    )


def get_model_param_from_user(clf_name):
    """
    Gets model parameters from the user based on the selected classifier.

    Parameters:
        clf_name (str): The name of the selected classifier.

    Returns:
        dict: A dictionary containing the user-selected model parameters.

    Displays the model parameter selection options using the Streamlit library.
    """
    params = {}
    st.sidebar.write("Select values: ")

    if clf_name == "Logistic Regression":
        R = st.sidebar.slider("Regularization", 0.1, 10.0, step=0.1)
        MI = st.sidebar.slider("max_iter", 50, 400, step=50)
        params["R"] = R
        params["MI"] = MI

    elif clf_name == "KNN":
        K = st.sidebar.slider("n_neighbors", 1, 20)
        params["K"] = K

    elif clf_name == "SVM":
        C = st.sidebar.slider("Regularization", 0.01, 10.0, step=0.01)
        kernel = st.sidebar.selectbox(
            "Kernel", ("linear", "poly", "rbf", "sigmoid", "precomputed")
        )
        params["C"] = C
        params["kernel"] = kernel

    elif clf_name == "Decision Trees":
        M = st.sidebar.slider("max_depth", 2, 20)
        C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        SS = st.sidebar.slider("min_samples_split", 1, 10)
        params["M"] = M
        params["C"] = C
        params["SS"] = SS

    elif clf_name == "Random Forest":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=100)
        M = st.sidebar.slider("max_depth", 2, 20)
        C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        params["N"] = N
        params["M"] = M
        params["C"] = C

    elif clf_name == "Gradient Boosting":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=100)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5)
        L = st.sidebar.selectbox("Loss", ("deviance", "exponential"))
        M = st.sidebar.slider("max_depth", 2, 20)
        params["N"] = N
        params["LR"] = LR
        params["L"] = L
        params["M"] = M

    elif clf_name == "XGBoost":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=50)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5, value=0.1)
        O = st.sidebar.selectbox(
            "Objective",
            ("binary:logistic", "reg:logistic", "reg:squarederror", "reg:gamma"),
        )
        M = st.sidebar.slider("max_depth", 1, 20, value=6)
        G = st.sidebar.slider("Gamma", 0, 10, value=5)
        L = st.sidebar.slider("reg_lambda", 1.0, 5.0, step=0.1)
        A = st.sidebar.slider("reg_alpha", 0.0, 5.0, step=0.1)
        CS = st.sidebar.slider("colsample_bytree", 0.5, 1.0, step=0.1)
        params["N"] = N
        params["LR"] = LR
        params["O"] = O
        params["M"] = M
        params["G"] = G
        params["L"] = L
        params["A"] = A
        params["CS"] = CS

    RS = 65
    params["RS"] = RS
    return params


def get_pred_inputs_from_user(data):
    """
    Gets user input for predicting the target variable based on the provided dataset.

    Parameters:
        data (pandas.DataFrame): The dataset used for input value references.

    Returns:
        dict: A dictionary containing the user-selected input values.

    Displays input fields for the user to enter their own values using the Streamlit library.
    """
    user_val = {}
    # User values
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("Try Your Own!")
    st.markdown(
        """
    In this section you can use your own values to predict the target variable. 
    Input the required values below and you will get your status based on the values. <br>
    <p style='color: red;'> 1 - High Risk </p> <p style='color: green;'> 0 - Low Risk </p>
    """,
        unsafe_allow_html=True,
    )
    for col in data.columns:
        name = col
        col = st.number_input(
            col,
            data[col].min(),
            data[col].max(),
        )
        user_val[name] = col

    return user_val
