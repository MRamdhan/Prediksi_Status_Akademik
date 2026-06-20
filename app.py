import streamlit as st
import numpy as np
import joblib


# ==========================
# LOAD MODEL
# ==========================

model = joblib.load(
    "model/random_forest_model.pkl"
)

scaler = joblib.load(
    "model/scaler.pkl"
)

encoder = joblib.load(
    "model/label_encoder.pkl"
)


# ==========================
# CONFIG
# ==========================

st.set_page_config(
    page_title="Prediksi Status Akademik",
    layout="wide"
)


# ==========================
# MENU
# ==========================

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Prediksi",
        "Dashboard",
        "Informasi"
    ]
)



# ==========================
# HALAMAN PREDIKSI
# ==========================

if menu == "Prediksi":

    st.title(
        "🎓 Prediksi Status Akademik Mahasiswa"
    )


    nama = st.text_input(
        "Nama Mahasiswa"
    )


    age = st.number_input(
        "Age at Enrollment",
        min_value=15,
        max_value=70
    )


    admission = st.number_input(
        "Admission Grade",
        min_value=0.0,
        max_value=200.0
    )


    tuition = st.selectbox(
        "Tuition Fees Up To Date",
        [
            0,
            1
        ]
    )


    st.subheader(
        "Semester 1"
    )


    sem1_enrolled = st.number_input(
        "Jumlah Mata Kuliah Semester 1",
        min_value=0
    )


    sem1_approved = st.number_input(
        "Jumlah Lulus Semester 1",
        min_value=0
    )


    sem1_grade = st.number_input(
        "Nilai Semester 1",
        min_value=0.0,
        max_value=20.0
    )



    st.subheader(
        "Semester 2"
    )


    sem2_enrolled = st.number_input(
        "Jumlah Mata Kuliah Semester 2",
        min_value=0
    )


    sem2_approved = st.number_input(
        "Jumlah Lulus Semester 2",
        min_value=0
    )


    sem2_grade = st.number_input(
        "Nilai Semester 2",
        min_value=0.0,
        max_value=20.0
    )



    # ==========================
    # PREDIKSI
    # ==========================


    if st.button("Prediksi"):


        # VALIDASI

        if sem1_approved > sem1_enrolled:

            st.error(
                "Jumlah mata kuliah Semester 1 yang lulus tidak boleh lebih besar dari yang diambil."
            )


        elif sem2_approved > sem2_enrolled:

            st.error(
                "Jumlah mata kuliah Semester 2 yang lulus tidak boleh lebih besar dari yang diambil."
            )


        else:


            # 36 FEATURES DATASET

            data = np.array([[
                0,0,0,0,0,0,0,0,0,0,0,0,
                admission,
                0,0,0,
                tuition,
                0,0,
                age,
                0,
                0,
                sem1_enrolled,
                0,
                0,
                sem1_approved,
                sem1_grade,
                0,
                sem2_enrolled,
                0,
                0,
                sem2_approved,
                sem2_grade,
                0,
                0,
                0
            ]])


            data_scaled = scaler.transform(
                data
            )


            prediction = model.predict(
                data_scaled
            )


            hasil = encoder.inverse_transform(
                prediction
            )[0]



            st.success(
                f"""
                Nama: {nama}

                Status Akademik:
                {hasil}
                """
            )



            if hasil == "Graduate":

                st.info(
                    "Mahasiswa diprediksi memiliki kemungkinan tinggi menyelesaikan studi."
                )


            elif hasil == "Enrolled":

                st.warning(
                    "Mahasiswa masih dalam proses pendidikan."
                )


            else:

                st.error(
                    "Mahasiswa memiliki risiko dropout."
                )



# ==========================
# DASHBOARD
# ==========================


elif menu == "Dashboard":


    st.title(
        "📊 Dashboard Hasil Penelitian"
    )


    knn = 70.17
    rf = 76.16


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Accuracy KNN",
            f"{knn}%"
        )


    with col2:

        st.metric(
            "Accuracy Random Forest",
            f"{rf}%"
        )



    st.divider()


    st.write(
        "Model terbaik:"
    )


    st.success(
        "Random Forest"
    )



    chart_data = {

        "KNN": knn,

        "Random Forest": rf

    }


    st.bar_chart(
        chart_data
    )



# ==========================
# INFORMASI
# ==========================


else:


    st.title(
        "Tentang Sistem"
    )


    st.write(
        """
        Sistem ini digunakan untuk melakukan prediksi
        status akademik mahasiswa menggunakan algoritma
        Random Forest berdasarkan data akademik.
        """
    )