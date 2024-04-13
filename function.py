import streamlit as st
import seaborn as sns
from googletrans import Translator
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def display_gpu_info(gpu_type):
    if gpu_type == "low performance":
        st.subheader("GPU dengan kategori low Performance")
        low_performance_info = [
            "- GPU dengan kategori low performance memiliki spesifikasi yang terbatas.",
            "- Biasanya cocok untuk tugas-tugas ringan seperti browsing web atau menonton video.",
            "- Kinerja grafisnya mungkin tidak cukup untuk permainan atau tugas komputasi yang membutuhkan daya pemrosesan yang tinggi.",
        ]
        st.markdown("\n".join(low_performance_info))
    elif gpu_type == "middle performance":
        st.subheader("GPU dengan kategori middle Performance")
        middle_performance_info = [
            "- GPU dengan kategori middle performance menawarkan keseimbangan antara kinerja dan harga.",
            "- Cocok untuk permainan dengan kualitas grafis menengah atau tugas-tugas komputasi sederhana hingga menengah.",
            "- Tidak sekuat GPU high performance, tetapi masih mampu menjalankan sebagian besar aplikasi yang membutuhkan grafis yang lebih baik.",
        ]
        st.markdown("\n".join(middle_performance_info))
    elif gpu_type == "high performance":
        st.subheader("GPU dengan kategori high Performance")
        high_performance_info = [
            "- GPU dengan kategori high performance memiliki spesifikasi yang tinggi dan kinerja grafis yang kuat.",
            "- Biasanya digunakan untuk permainan dengan kualitas grafis tinggi, rendering video, atau tugas komputasi berat lainnya.",
            "- Meskipun memiliki harga yang lebih tinggi, GPU ini menawarkan kinerja terbaik dalam hal grafis dan pemrosesan.",
        ]
        st.markdown("\n".join(high_performance_info))


def scatter_plot(df):
    st.subheader(
        """
1. Distribusi data berdasarkan: Skor G3Dmark dan Harga GPU
"""
    )
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    # Membuat scatter plot
    sns.scatterplot(
        data=df,
        x="G3Dmark",
        y="price",
        hue="PerformanceCategory",
        palette="viridis",
        ax=ax,
    )

    ax.set_xlabel("G3Dmark")
    ax.set_ylabel("Price")
    ax.set_title("Distribusi data berdasarkan: Skor G3Dmark dan Harga (Price)")
    ax.grid(True)
    ax.legend(title="Performance Category")

    st.pyplot(fig)
    st.write(
        """
Dari visualisasi plot diatas, kita bisa mengamati beberapa hal:

- Ada kluster titik di area harga rendah dengan nilai G3D mark rendah, yang menunjukkan banyak komponen memiliki kinerja rendah dan harga murah (low p).
- Sebagian titik, terutama yang berkategori 2 (high p), memiliki nilai G3D mark yang tinggi, sebagaimana digambarkan pada sumbu horizontal yang jauh ke kanan, menunjukkan bahwa ini adalah komponen kinerja tinggi.
- Komponen kinerja tinggi (kategori high p) juga cenderung memiliki harga yang lebih tinggi.
- Terdapat beberapa outlier di mana beberapa komponen dengan harga sangat tinggi tidak memiliki skor G3D mark yang tinggi sebanding dengan harga mereka.

"""
    )


def scatter_plot2(df):
    st.subheader(
        """
2. Distribusi data berdasarkan: Skor G3Dmark dan Daya TDP
"""
    )
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    # Membuat scatter plot
    sns.scatterplot(
        data=df,
        x="G3Dmark",
        y="TDP",
        hue="PerformanceCategory",
        palette="viridis",
        ax=ax,
    )

    ax.set_xlabel("G3Dmark")
    ax.set_ylabel("TDP (performance per watt)")
    ax.set_title("Distribusi data berdasarkan: Skor G3Dmark dan Daya TDP")
    ax.grid(True)
    ax.legend(title="Performance Category")

    st.pyplot(fig)
    st.write(
        """
Berdasarkan visualisasi diatas, kita bisa melihat bahwa terdapat kecenderungan bahwa semakin tinggi Skor G3DMark, yang mengindikasikan performa grafis yang lebih baik, maka biasanya diperlukan TDP yang lebih tinggi. 
- Kategori performa 2 (high p), yang biasanya memiliki Skor G3DMark tinggi, kebanyakan terdapat pada TDP yang tinggi. 
- Sebaliknya, kategori performa 0 (low p) yang memiliki Skor G3DMark lebih rendah, cenderung memiliki nilai TDP yang lebih rendah. 
- Kategori performa 1 (middle p) terletak di antara kedua kategori lainnya, dengan variasi TDP dan Skor G3DMark yang lebih tersebar.
"""
    )


def heatmap(df):
    df2 = df.drop(["gpuName"], axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df2.corr(), annot=True, cmap="coolwarm", ax=ax)
    plt.title("Korelasi berdasarkan fitur numerik")
    st.pyplot(fig)
    st.write(
        """
Berikut ini adalah beberapa hasil observasi yang didapatkan dari korelasi berdasarkan heatmap:

- GD3mark dan PerformanceCategory: Terdapat korelasi negatif cukup kuat antara GD3mark dan PerformanceCategory, seperti yang ditunjukkan oleh warna biru dan nilai koefisien sebesar -0.84.
- Harga dan PerformanceCategory: Tampak ada korelasi negatif lemah antara harga dan PerformanceCategory dengan koefisien korelasi -0.25.
- GD3mark dan TDP: Terlihat ada hubungan positif yang sedang antara GD3mark dan TDP dengan koefisien korelasi sebesar 0.72.
- Korelasi yang paling lemah: Antara harga dan testDate, ada korelasi yang sangat lemah yang hampir tidak signifikan, ditandai dengan nilai 0.01.
- Kategori dan TDP: Ada korelasi negatif yang kuat antara kategori dan TDP dengan nilai -0.77.
"""
    )


def compositionAndComparison(df):

    df["PerformanceCategory"].replace(
        {2: "low performance", 1: "middle performance", 0: "high performance"},
        inplace=True,
    )

    df_numeric = df.drop(columns=["gpuName"])

    class_composition = df_numeric.groupby("PerformanceCategory").mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(class_composition.T, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Komposisi dan komparasi berdasarkan tiap kategori performa GPU")
    plt.xlabel("Class")
    plt.ylabel("Feature")
    st.pyplot(plt)

    st.write(
        """
- Dapat disimpulkan bahwa GPU dengan high performance class memiliki feature size yang jauh lebih besar dari kelas lainnya dan harganya juga yang paling mahal. 
- Untuk midrange class, feature size dan harganya lebih rendah dibandingkan high performance dan lebih tinggi jika dibandingkan dengan kategori low performance. 
- Sedangkan low performance class, skor 3Dmark, 2Dmark, harga, dan TDPnya merupakan yang terendah dari kedua kategori sebelumnya.
"""
    )


def predict():
    # gpuName = st.selectbox("Select GPU",[i for i in df['gpuName'].unique()])
    G3Dmark = st.number_input("G3Dmark", min_value=0)
    G2Dmark = st.number_input("G2Dmark", min_value=0)
    price = st.number_input("Price", min_value=0.0)
    TDP = st.number_input("TDP", min_value=0.0)
    testDate_year = st.number_input("Test Date Year", min_value=2018, max_value=2023)
    category = st.number_input("Category", min_value=0, max_value=2)

    # Membuat DataFrame user_data
    user_data = pd.DataFrame(
        {
            "G3Dmark": [G3Dmark],
            "G2Dmark": [G2Dmark],
            "price": [price],
            "TDP": [TDP],
            "testDate": [testDate_year],
            "category": [category],
        }
    )

    st.subheader("Data Pengguna:")
    st.write(user_data)

    button = st.button("Predict")
    if button:
        with open("gnb.pkl", "rb") as file:
            loaded_model = pickle.load(file)
        predicted_performance_category = loaded_model.predict(user_data)

        if predicted_performance_category == 0:
            st.write("Predicted Performance Category: high Performance")
        elif predicted_performance_category == 1:
            st.write("Predicted Performance Category: Middle Performance")
        elif predicted_performance_category == 2:
            st.write("Predicted Performance Category: low Performance")
