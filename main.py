import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from streamlit_option_menu import *
from function import *


df = pd.read_csv("Data_Cleanednewest.csv")
dfbm = pd.read_csv("Data Cleaned-gpubench.csv")  # Before mapping

st.title(
    "Analisa Data Skor Benchmark Pada GPU Komputer Sebagai Bahan Acuan User Dalam Memilih GPU Yang Sesuai Kebutuhan"
)
st.write("Dataframe: ", df)

with st.sidebar:
    selected = option_menu(
        "Kategori GPU berdasarkan skor G3Dmark",
        [
            "Introducing Data",
            "Data Distribution",
            "Relation",
            "Composition & Comparison",
            "Predict",
        ],
        default_index=0,
    )

if selected == "Introducing Data":
    st.title("Kategori GPU berdasarkan performanya: ")
    st.write(
        """
    Kategori GPU tersebut didapatkan berdasarkan skor benchmark render 3D menggunakan software 3DMark, semakin tinggi performanya maka semakin baik pula kinerjanya, namun dibalik tingginya performa tersebut juga ada faktor lain seperti harga dan efisiensi daya yang digunakan.
    """
    )

    gpu_types = ["low performance", "middle performance", "high performance"]
    gpu_type = st.selectbox(
        "Pilihlah Tipe GPU berdasarkan performanya, agar mengetahui insightnya: ",
        gpu_types,
    )

    display_gpu_info(gpu_type)

if selected == "Data Distribution":
    st.header("Data Distribution")
    scatter_plot(df)
    scatter_plot2(df)

if selected == "Relation":
    st.title("Relations")
    heatmap(df)

if selected == "Composition & Comparison":
    st.title("Composition")
    compositionAndComparison(df)

if selected == "Predict":
    st.title("Mencoba Memprediski Kategori GPU: ")
    predict()
