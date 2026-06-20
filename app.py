from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model/random_forest_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/dashboard")
def dashboard():
    knn_accuracy = 70.17
    rf_accuracy = 76.16

    if rf_accuracy > knn_accuracy:
        best_model = "Random Forest"
    else:
        best_model = "KNN"

    return render_template(
        "dashboard.html",
        knn=knn_accuracy,
        rf=rf_accuracy,
        best=best_model
    )
    
@app.route("/informasi")
def informasi():
    return render_template("informasi.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        nama = request.form["nama"]
        age = float(request.form["age"])
        admission = float(request.form["admission"])
        tuition = int(request.form["tuition"])
        sem1_enrolled = int(request.form["sem1_enrolled"])
        sem1_approved = int(request.form["sem1_approved"])
        sem1_grade = float(request.form["sem1_grade"])
        sem2_enrolled = int(request.form["sem2_enrolled"])
        sem2_approved = int(request.form["sem2_approved"])
        sem2_grade = float(request.form["sem2_grade"])

        # ==========================
        # VALIDASI INPUT
        # ==========================
        if sem1_approved > sem1_enrolled:
            return render_template(
                "index.html",
                error="Jumlah mata kuliah Semester 1 yang lulus tidak boleh lebih besar dari mata kuliah yang diambil."
            )

        if sem2_approved > sem2_enrolled:
            return render_template(
                "index.html",
                error="Jumlah mata kuliah Semester 2 yang lulus tidak boleh lebih besar dari mata kuliah yang diambil."
            )

        if sem1_grade < 0 or sem1_grade > 20:
            return render_template(
                "index.html",
                error="Nilai Semester 1 harus berada pada rentang 0 - 20."
            )

        if sem2_grade < 0 or sem2_grade > 20:
            return render_template(
                "index.html",
                error="Nilai Semester 2 harus berada pada rentang 0 - 20."
            )

        # ==========================
        # DATA 36 FITUR
        # ==========================
        data = np.array([[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            admission,
            0, 0, 0,
            tuition,
            0, 0,
            age,
            0, 0,
            sem1_enrolled, 0, sem1_approved, sem1_grade, 0, 0,
            sem2_enrolled, 0, sem2_approved, sem2_grade, 0, 0, 0, 0
        ]])

        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        hasil = encoder.inverse_transform(prediction)[0]

        # Keterangan hasil
        if hasil == "Graduate":
            keterangan = "Mahasiswa diprediksi memiliki kemungkinan tinggi untuk menyelesaikan studi."
        elif hasil == "Enrolled":
            keterangan = "Mahasiswa masih dalam proses pendidikan."
        else:
            keterangan = "Mahasiswa memiliki risiko berhenti studi."

        return render_template(
            "index.html",
            nama=nama,
            prediction=hasil,
            keterangan=keterangan
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)