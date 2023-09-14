import pandas as pd
import streamlit as st


def load_data():
    """
    Loads the heart attack dataset.

    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    data = pd.read_csv("Data/heart.csv")
    return data


def preprocess_data(data):
    """
        Preprocesses the heart attack dataset.

        Parameters:
            data (pandas.DataFrame): The dataset to preprocess.

        Returns:
            pandas.DataFrame: The preprocessed features (X) and target (Y).
    .
    """
    X = data.drop(["output"], axis=1)
    Y = data.output

    st.header("Dataset")
    with st.expander("Show Dataset"):
        st.write(data)

    st.write("Shape of dataset: ", data.shape)
    st.write("Number of classes: ", Y.nunique())

    return X, Y
