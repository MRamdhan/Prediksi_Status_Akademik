import joblib
import numpy as np
import streamlit as st

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Prediksi Status Akademik", layout="wide")

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/label_encoder.pkl")

# ==========================
# MENU
# ==========================
menu = st.sidebar.selectbox("Menu", ["Prediksi", "Dashboard", "Informasi"])

# ==========================
# HALAMAN PREDIKSI
# ==========================
if menu == "Prediksi":
    st.title("🎓 Prediksi Status Akademik Mahasiswa")

    nama = st.text_input("Nama Mahasiswa")
    age = st.number_input("Umur Saat Masuk Kuliah", min_value=15, max_value=70)
    admission = st.number_input("Nilai Masuk Kuliah", min_value=0.0, max_value=200.0)
    tuition_status = st.selectbox(
        "Status Pembayaran UKT",
        [
            "Sudah Bayar",
            "Belum Bayar"
        ]
    )


    if tuition_status == "Sudah Bayar":
        tuition = 1
    else:
        tuition = 0

    st.subheader("Semester 1")
    sem1_enrolled = st.number_input("Jumlah Mata Kuliah Semester 1", min_value=0)
    sem1_approved = st.number_input("Jumlah Lulus Semester 1", min_value=0)
    sem1_grade = st.number_input("Nilai Semester 1 (Skala Dataset 0-20)", min_value=0.0, max_value=20.0)

    st.subheader("Semester 2")
    sem2_enrolled = st.number_input("Jumlah Mata Kuliah Semester 2", min_value=0)
    sem2_approved = st.number_input("Jumlah Lulus Semester 2", min_value=0)
    sem2_grade = st.number_input("Nilai Semester 2 (Skala Dataset 0-20)", min_value=0.0, max_value=20.0)

    # ==========================
    # PREDIKSI
    # ==========================
    if st.button("Prediksi"):
        # VALIDASI
        if sem1_approved > sem1_enrolled:
            st.error("Jumlah mata kuliah Semester 1 yang lulus tidak boleh lebih besar dari yang diambil.")
        elif sem2_approved > sem2_enrolled:
            st.error("Jumlah mata kuliah Semester 2 yang lulus tidak boleh lebih besar dari yang diambil.")
        else:
            # 36 FEATURES DATASET
            data = np.array([[
                0, # Marital status
                0, # Application mode
                0, # Application order
                0, # Course
                0, # Daytime
                0, # Previous qualification
                0, # Previous qualification grade
                0, # Nationality
                0, # Mother qualification
                0, # Father qualification
                0, # Mother occupation
                0, # Father occupation
                admission, # Admission grade
                0, # Displaced
                0, # Educational special needs
                0, # Debtor
                tuition, # Tuition fees up to date
                0, # Gender
                0, # Scholarship holder
                age, # Age enrollment
                0, # International
                0, # sem1 credited
                sem1_enrolled, # sem1 enrolled
                0, # sem1 evaluations
                sem1_approved, # sem1 approved
                sem1_grade, # sem1 grade
                0, # sem1 without evaluation
                0, # sem2 credited
                sem2_enrolled, # sem2 enrolled
                0, # sem2 evaluation
                sem2_approved, # sem2 approved
                sem2_grade, # sem2 grade
                0, # sem2 without evaluation
                0, # unemployment
                0, # inflation
                0  # GDP
            ]])

            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)
            prob = model.predict_proba(data_scaled)
            hasil = encoder.inverse_transform(prediction)[0]
            st.write(
                {
                    "Dropout": prob[0][0],
                    "Enrolled": prob[0][1],
                    "Graduate": prob[0][2]
                }
            )

            st.success(f"""
            Nama: {nama}
            
            Status Akademik: {hasil}
            """)

            if hasil == "Graduate":
                st.info("Mahasiswa diprediksi memiliki kemungkinan tinggi menyelesaikan studi.")
            elif hasil == "Enrolled":
                st.warning("Mahasiswa masih dalam proses pendidikan.")
            else:
                st.error("Mahasiswa memiliki risiko dropout.")

# ==========================
# DASHBOARD
# ==========================
elif menu == "Dashboard":
    st.title("📊 Dashboard Hasil Penelitian")

    knn = 70.17
    rf = 76.16

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy KNN", f"{knn}%")
    with col2:
        st.metric("Accuracy Random Forest", f"{rf}%")

    st.divider()
    st.write("Model terbaik:")
    st.success("Random Forest")

    chart_data = {"KNN": knn, "Random Forest": rf}
    st.bar_chart(chart_data)

# ==========================
# INFORMASI
# ==========================
else:
    st.title("Tentang Sistem")
    st.write("""
    Sistem ini digunakan untuk melakukan prediksi status akademik mahasiswa 
    menggunakan algoritma Random Forest berdasarkan data akademik.
    """)