# 🌾 Crop Yield Prediction Web Application

An end-to-end Machine Learning web application that predicts crop yield based on environmental and soil parameters, helping enable data-driven agricultural decisions.

🔗 **Live Demo:** https://crop-yield-prediction-te69.onrender.com
⚠️ *(App may take 30–60 seconds to wake up on first load)*

---

## 📌 Overview

This application predicts crop yield using key agricultural factors such as:

* 🌧 Rainfall
* 🌡 Temperature
* 💧 Humidity
* 🌱 Soil nutrients (N, P, K)
* ⚗ Soil pH
* 🌾 Crop type and season

It also provides:

* 📊 Estimated yield output
* 💰 Revenue estimation
* 📈 Basic advisory insights

---

## 🚀 Key Features

* ✅ Real-time crop yield prediction
* ✅ End-to-end ML pipeline (training → deployment)
* ✅ Interactive web interface using Flask
* ✅ Cloud deployment (Render) for public access
* ✅ Clean and user-friendly UI

---

## 🧠 Machine Learning

* **Algorithm:** Random Forest Regressor
* **Train-Test Split:** 80/20
* **R² Score:** *0.XX (update with your actual score)*

### Feature Engineering:

* One-hot encoding
* Feature scaling
* Data preprocessing

---

## ⚙️ How It Works

1. User inputs environmental and soil data
2. Flask backend processes the input
3. Trained ML model predicts crop yield
4. Results are displayed with insights

---

## 📊 Dataset

This project uses a realistic synthetic agricultural dataset simulating:

* 28 Indian states
* Multiple crops
* Seasonal variations
* Soil and climate conditions

*(Dataset created for ML experimentation purposes)*

---

## 🛠 Tech Stack

* **Programming:** Python
* **Backend:** Flask
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Frontend:** HTML, CSS, Bootstrap
* **Deployment:** Render

---

## 📸 Application Preview

*(Add screenshots here for better visibility)*
Example:
![App Screenshot](screenshots/ui.png)

---

## 💻 Run Locally

```bash
# Clone repository
git clone https://github.com/pravalika-mekala/crop-yield-prediction.git

# Navigate to project folder
cd crop-yield-prediction

# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Run application
python app.py
```

Open in browser:
👉 http://127.0.0.1:5000

---

## 🎯 Purpose

This project demonstrates the application of machine learning in agriculture, enabling smarter crop planning and productivity optimization using data-driven insights.

---

## 👩‍💻 Author

**Pravalika Mekala**
CSE (Data Science) Student

🔗 GitHub: https://github.com/pravalika-mekala
🔗 LinkedIn: https://www.linkedin.com/in/mekala-pravalika-12a774374

---

⭐ If you found this project useful, consider giving it a star!
