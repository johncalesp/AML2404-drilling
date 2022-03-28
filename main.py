import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import time
import matplotlib.pyplot as plt
import seaborn as sns
from os import path

st.write("""
# Oilfield Coring
The following graph is a visual aid about how the predictions, made by the model, will look to an operator
when they are drilling.
""")

def generate_data():
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "data", "features.csv"))
    X = pd.read_csv(filepath)

    return X.sample(100)

@st.cache
def load_model():
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "data", "catboost_model_resampled"))
    local_model = CatBoostClassifier()  # parameters not required.
    local_model.load_model(filepath)
    return local_model


model = load_model()
data = generate_data()

def simulation():
    t = st.empty()
    pl = st.empty()
    fig, ax = plt.subplots()
    sns.set()
    for idx in range(len(data)):
        t.write(pd.DataFrame(data.values[idx].reshape(1,-1)))
        time.sleep(2)
        y_pred = model.predict(data.values[idx])
        if y_pred == 0: color="green"
        else: color = "red"
        sns.scatterplot(x=[idx],y=[y_pred+1],color=color,s=100)
        ax.set_title("Drilling Indication by time")
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("Drilling (Yes/No)")
        pl.pyplot(fig)


if st.button('Simulation'):
     simulation()
else:
     st.write('Press button to Start simulation')

