# VisionTrack-AI-Face-Recognition-Attendance-System
Engineered with Python, Streamlit, OpenCV, and Scikit-Learn (KNN), this web app provides real-time face matching. It features secure employee self-registration, role-based dashboards, automated daily check-ins, visual attendance graphs, and dynamic monthly absentee warnings.

Here is a clean, professional `README.md` file tailored specifically for your GitHub repository.

---

# 👁️ VisionTrack

### **AI-Powered Face Recognition Attendance System**

**VisionTrack** is a high-performance web application engineered with **Python**, **Streamlit**, and **OpenCV** to automate attendance tracking. It uses a **K-Nearest Neighbors (KNN)** machine learning model for real-time face matching and features a modern **Glassmorphism UI**.

---

## 🚀 Key Features

* **Self-Registration:** Employees can securely register their own face, email, and password.
* **Dual Dashboards:** * **Admin:** Manage employees, view global attendance (Present/Absent status), and monitor monthly warnings.
* **Employee:** Mark attendance via webcam, view personal history, and track lifetime statistics.


* **Smart Analytics:** Automated calculation of "Present vs. Absent" days starting from the specific registration timestamp.
* **Absentee Warnings:** Dynamic system alerts if an employee exceeds 3 absences in the current month.
* **Visual Insights:** Integrated small-scale bar charts for quick attendance overviews.
* **Security:** Multi-file synchronization (`JSON` based) ensuring face embeddings and user data are always matched.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Custom Glassmorphism CSS)
* **Computer Vision:** OpenCV (Haar Cascades)
* **Machine Learning:** Scikit-Learn (K-Nearest Neighbors)
* **Data Handling:** NumPy, JSON, PIL

---

## 📦 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/VisionTrack.git
cd VisionTrack

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```

3. **Run the application:**
```bash
streamlit run faceapp.py

```

---

## 📁 Project Structure

* `faceapp.py`: Main application logic and UI.
* `users.json`: Stores employee profile data and credentials.
* `embeddings.json`: Stores flattened face feature vectors for the AI model.
* `attendance.json`: Logs every successful check-in with date and time.
* `requirements.txt`: List of necessary Python packages.

---

## 🔑 Credentials

* **Default Admin Password:** `admin123`
* **Employee Login:** Use your registered Name and Password.
