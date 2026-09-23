import streamlit as st
import cv2
import numpy as np
import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# =========================
# FILE PATHS
# =========================
USERS_FILE = "users.json"
EMBEDDINGS_FILE = "embeddings.json"
ATTENDANCE_FILE = "attendance.json"
ADMIN_FILE = "admin.json"

# =========================
# INIT FILES
# =========================
for file, default_data in [
    (USERS_FILE, {}),
    (EMBEDDINGS_FILE, {}),
    (ATTENDANCE_FILE, []),
    (ADMIN_FILE, {"admin": "admin123"})
]:
    if not os.path.exists(file):
        with open(file, "w") as f:
            json.dump(default_data, f)

# =========================
# LOAD/SAVE FUNCTIONS
# =========================
def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f)

def load_embeddings():
    with open(EMBEDDINGS_FILE, "r") as f:
        return json.load(f)

def save_embeddings(data):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(data, f)

def load_attendance():
    with open(ATTENDANCE_FILE, "r") as f:
        return json.load(f)

def save_attendance(data):
    with open(ATTENDANCE_FILE, "w") as f:
        json.dump(data, f)

def load_admin():
    with open(ADMIN_FILE, "r") as f:
        return json.load(f)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Vision - Face Recognition Attendance",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# GLASSMORPHISM CSS & THEMES
# =========================
st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    min-height: 100vh;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
}

.main {
    background: transparent !important;
}

.stMainBlockContainer {
    background: transparent !important;
    padding-top: 2rem;
}

/* GLASSMORPHISM CONTAINER */
.glass-container {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    margin: 20px 0;
    color: white;
}

.glass-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.25);
    color: white;
    margin: 15px 0;
}

/* BUTTONS */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3.5em;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.4) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(31, 38, 135, 0.6) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* INPUT FIELDS */
.stTextInput > div > div > input,
.stPasswordInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
    background: rgba(255, 255, 255, 0.6) !important;
    color: #111 !important; 
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 10px !important;
    padding: 12px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
}

.stTextInput > div > div > input::placeholder,
.stPasswordInput > div > div > input::placeholder {
    color: #222222 !important; 
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* TEXT & HEADINGS */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

p, label, span {
    color: rgba(255, 255, 255, 0.95) !important;
}

/* SUCCESS/ERROR/WARNING MESSAGES */
.stAlert {
    background: rgba(255, 255, 255, 0.15) !important;
    border-left: 4px solid;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
}

.stSuccess {
    border-left-color: #10b981 !important;
}

.stError {
    border-left-color: #ef4444 !important;
}

.stWarning {
    border-left-color: #f59e0b !important;
}

.stInfo {
    border-left-color: #3b82f6 !important;
}

/* TITLE STYLES */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 900;
    background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
    text-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.subtitle {
    text-align: center;
    color: rgba(255, 255, 255, 0.9);
    margin-bottom: 30px;
    font-size: 18px;
    font-weight: 500;
}

.metric-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    color: white;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #ffd89b;
}

.metric-label {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.8);
    margin-top: 8px;
}

/* DATAFRAME */
.stDataFrame {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
}

.stDataFrame > div {
    background: transparent !important;
}

/* CAMERA INPUT */
.stCameraInput > div > div > button {
    border-radius: 10px !important;
}

/* RADIO BUTTONS & CHECKBOXES */
.stRadio > div,
.stCheckbox > div {
    color: white !important;
}

/* DIVIDER */
hr {
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* LOGIN PAGE SPECIFIC */
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
}

.login-box {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 25px;
    padding: 50px;
    width: 100%;
    max-width: 450px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

/* TABS - CENTERED ALIGNMENT FIX */
div[data-baseweb="tab-list"] {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    gap: 10px !important;
}

.stTabs > div > div > button {
    background: transparent !important;
    color: rgba(255, 255, 255, 0.7) !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    flex: none !important; 
}

.stTabs > div > div > button[aria-selected="true"] {
    color: white !important;
    border-bottom-color: #ffd89b !important;
}

/* COLUMNS */
[data-testid="column"] {
    background: transparent !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.5);
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.user_id = None
    st.session_state.emp_name = ""
if "cam_key" not in st.session_state:
    st.session_state.cam_key = 0

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# EXTRACT EMBEDDING
# =========================
def extract_embedding(face):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # 1. LIGHTING FIX: Histogram Equalization balances shadows and bright spots
        gray = cv2.equalizeHist(gray) 
        
        # 2. DETAIL FIX: Increase resolution to 128x128 for more unique features
        resized = cv2.resize(gray, (200, 200))
        
        # Flatten and Normalize
        embedding = resized.flatten().astype("float32")
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        embedding = embedding / norm
        return embedding.tolist()
    except Exception as e:
        return None

# =========================
# TRAIN KNN
# =========================
# =========================
# TRAIN KNN
# =========================
def train_knn():
    embeddings = load_embeddings()
    if len(embeddings) == 0:
        return None
        
    X = []
    y = []
    for uid, emb in embeddings.items():
        X.append(emb)
        y.append(uid)
        
    # FIX: n_neighbors MUST be exactly 1 because we only have 1 picture per user
    knn = KNeighborsClassifier(
        n_neighbors=1,
        metric='euclidean'
    )
    
    knn.fit(X, y)
    return knn

# =========================
# MARK ATTENDANCE
# =========================
def mark_attendance(user_id, name):
    data = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    for entry in data:
        if entry["user_id"] == user_id and entry["date"] == today:
            return False
    data.append({
        "user_id": user_id,
        "name": name,
        "date": today,
        "time": current_time
    })
    save_attendance(data)
    return True

# =========================
# LOGIN PAGE
# =========================
def login_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("""
        <div class="glass-container" style="text-align: center;">
            <div class="title">👁️ VisionTrack</div>
            <div class="subtitle">Face Recognition Attendance System</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["👤 Login", "📝 Register", "🔐 Admin"])
        
        with tab1:
            st.markdown("""
            <div class="glass-card">
                <h3 style="text-align: center; margin-bottom: 20px;">Employee Login</h3>
            </div>
            """, unsafe_allow_html=True)
            
            emp_name = st.text_input("📝 Enter Your Name", key="login_emp_name", placeholder="Your full name")
            emp_password = st.text_input("🔑 Enter Password", type="password", key="emp_pass", placeholder="Your password")
            
            if st.button("🚀 Login as Employee", key="emp_login"):
                if emp_name == "" or emp_password == "":
                    st.error("❌ Please fill all fields")
                else:
                    users = load_users()
                    
                    user_found = False
                    password_correct = False
                    
                    for u in users.values():
                        if u["name"].lower() == emp_name.lower():
                            user_found = True
                            if u.get("password", "") == emp_password:
                                password_correct = True
                            break
                    
                    if user_found and password_correct:
                        st.session_state.logged_in = True
                        st.session_state.user_role = "employee"
                        st.session_state.emp_name = emp_name
                        st.rerun()
                    elif user_found and not password_correct:
                        st.error("❌ Incorrect password!")
                    else:
                        st.error("❌ Employee not found! Please register first.")
        
        with tab2:
            st.markdown("""
            <div class="glass-card">
                <h3 style="text-align: center; margin-bottom: 20px;">Employee Self-Registration</h3>
            </div>
            """, unsafe_allow_html=True)
            
            new_emp_name = st.text_input("👤 Your Full Name", key="reg_name", placeholder="Enter full name")
            new_emp_email = st.text_input("📧 Your Email", key="reg_email", placeholder="Enter email")
            new_emp_password = st.text_input("🔑 Create Password", type="password", key="reg_pass", placeholder="Create a password")
            
            picture = st.camera_input("📸 Capture Your Face", key=f"emp_cam_{st.session_state.cam_key}")
            
            if st.button("✅ Register My Face", key="emp_register_btn"):
                if new_emp_name == "":
                    st.error("❌ Please enter your name")
                elif new_emp_password == "":
                    st.error("❌ Please create a password")
                elif picture is None:
                    st.error("❌ Please capture an image")
                else:
                    users = load_users()
                    
                    if any(u["name"].lower() == new_emp_name.lower() for u in users.values()):
                        st.error("❌ A user with this name is already registered!")
                    else:
                        image = Image.open(picture)
                        image = np.array(image)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                        
                        if len(faces) == 0:
                            st.error("❌ No face detected in the image. Please try again with good lighting.")
                        else:
                            x, y, w, h = faces[0]
                            face_crop = image[y:y+h, x:x+w]
                            embedding = extract_embedding(face_crop)
                            
                            if embedding is None:
                                st.error("❌ Failed to extract face features")
                            else:
                                user_id = str(uuid.uuid4())
                                embeddings = load_embeddings()
                                
                                users[user_id] = {
                                    "name": new_emp_name,
                                    "email": new_emp_email,
                                    "password": new_emp_password, 
                                    "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                embeddings[user_id] = embedding
                                
                                save_users(users)
                                save_embeddings(embeddings)
                                
                                st.success(f"✅ Registered {new_emp_name} successfully! You can now log in.")
                                st.session_state.cam_key += 1
                                st.rerun()

        with tab3:
            st.markdown("""
            <div class="glass-card">
                <h3 style="text-align: center; margin-bottom: 20px;">Admin Login</h3>
            </div>
            """, unsafe_allow_html=True)
            
            admin_password = st.text_input("🔑 Admin Password", type="password", key="admin_pass", placeholder="Enter admin password")
            
            if st.button("🔓 Login as Admin", key="admin_login"):
                if admin_password == "":
                    st.error("❌ Please enter password")
                elif admin_password == "admin123":
                    st.session_state.logged_in = True
                    st.session_state.user_role = "admin"
                    st.rerun()
                else:
                    st.error("❌ Invalid admin password")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        

# =========================
# EMPLOYEE DASHBOARD
# =========================
def employee_dashboard():
    col1, col2, col3 = st.columns([0.8, 2, 0.8])
    
    with col2:
        st.markdown("""
        <div class="glass-container" style="text-align: center;">
            <div class="title">👁️ VisionTrack</div>
            <div class="subtitle">Welcome, """ + st.session_state.get("emp_name", "Employee")+ """</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚪 Logout", key="emp_logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📸 Mark Attendance", "📋 My Records", "ℹ️ About"])
    
    with tab1:
        st.markdown("""
        <div class="glass-card">
            <h2>📸 Scan Your Face</h2>
            <p>Capture your face to mark attendance</p>
        </div>
        """, unsafe_allow_html=True)
        
        knn = train_knn()
        
        if knn is None:
            st.warning("⚠️ No registered employees found. Please contact admin.")
        else:
            picture = st.camera_input("📷 Capture Your Face", key=f"mark_cam_{st.session_state.cam_key}")
            
            if picture is not None:
                image = Image.open(picture)
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) == 0:
                    st.error("❌ No face detected. Please try again.")
                else:
                    users = load_users()
                    recognized = False
                    
                    for (x, y, w, h) in faces:
                        face_crop = image[y:y+h, x:x+w]
                        embedding = extract_embedding(face_crop)
                        
                        if embedding is None:
                            continue
                        
                        prediction = knn.predict([embedding])[0]
                        pred_id = str(prediction)
                        distance, _ = knn.kneighbors([embedding])
                        distance = distance[0][0]
                        
                        if distance < 0.65:
                            if pred_id in users: 
                                name = users[pred_id]["name"]
                                current_user = st.session_state.get("emp_name", "")
                                if name.lower() == current_user.lower():
                                    marked = mark_attendance(pred_id, name)
                                    color = (0, 255, 0)
                                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
                                    cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                                    
                                    if marked:
                                        st.success(f"✅ Attendance marked for {name}")
                                        st.session_state.cam_key += 1 
                                        recognized = True
                                    else:
                                        st.warning(f"⚠️ {name} already marked today")
                                        recognized = True
                                else:
                                    st.error(f"❌ Face matches {name}, but you logged in as {current_user}.")
                                    recognized = True
                            else:
                                st.error("❌ Face recognized, but user data is missing (database out of sync).")
                                recognized = True
                        else:
                            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
                            cv2.putText(image, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if not recognized:
                        st.error("❌ Face not recognized or didn't match login. Please try again.")
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image, caption="🖼️ Recognition Result", use_container_width=True)
    
    # ===== NEW TAB: MY RECORDS (COUNTERS & GRAPH) =====
    with tab2:
        attendance = load_attendance()
        users = load_users()
        current_user = st.session_state.get("emp_name", "")
        
        current_uid = next((uid for uid, info in users.items() if info["name"].lower() == current_user.lower()), None)
                
        if current_uid:
            reg_date_str = users[current_uid].get("registered_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            reg_date = datetime.strptime(reg_date_str.split(" ")[0], "%Y-%m-%d").date()
            today = datetime.now().date()
            
            # 1. CALCULATE TOTALS FROM REGISTERED DATE
            all_dates = [reg_date + timedelta(days=x) for x in range((today - reg_date).days + 1)]
            present_dates = {a["date"] for a in attendance if a["user_id"] == current_uid}
            
            total_days = len(all_dates)
            present_count = len([d for d in all_dates if d.strftime("%Y-%m-%d") in present_dates])
            absent_count = total_days - present_count

            # 2. MONTHLY WARNING LOGIC
            current_month = today.strftime("%Y-%m")
            days_this_month = [d for d in all_dates if d.strftime("%Y-%m") == current_month]
            present_this_month = [d for d in days_this_month if d.strftime("%Y-%m-%d") in present_dates]
            absent_this_month = len(days_this_month) - len(present_this_month)

            if absent_this_month > 3:
                st.error(f"⚠️ **ATTENTION:** You have {absent_this_month} absences this month. Please maintain regular attendance to avoid disciplinary action.")

            # 3. DISPLAY METRICS
            st.markdown(f"**Registration Date:** `{reg_date_str}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Days", total_days)
            c2.metric("Total Present", present_count, delta_color="normal")
            c3.metric("Total Absent", absent_count, delta_color="inverse")

            # 4. ATTENDANCE GRAPH
            st.markdown("### 📊 Attendance Overview")
            chart_data = {"Status": ["Present", "Absent"], "Days": [present_count, absent_count]}
            
            # Use columns and height to make the graph smaller
            graph_col, empty_col = st.columns([1, 2]) # 1 part graph, 2 parts empty space
            with graph_col:
                st.bar_chart(chart_data, x="Status", y="Days", height=200)

            # 5. DATE PICKER SEARCH
            st.markdown("---")
            check_date = st.date_input("🗓️ Check Specific Date", value=today, max_value=today, min_value=reg_date)
            check_date_str = check_date.strftime("%Y-%m-%d")
            
            record = next((a for a in attendance if a["user_id"] == current_uid and a["date"] == check_date_str), None)
            if record:
                st.success(f"✅ Present on {check_date_str} at {record['time']}")
            else:
                st.error(f"❌ Absent on {check_date_str}")
        else:
            st.error("❌ User data not found.")

    with tab3:
        st.markdown("""
        <div class="glass-card">
            <h3>📖 About VisionTrack</h3>
            <p>VisionTrack is an AI-powered face recognition attendance system that uses advanced computer vision technology to automatically mark your attendance.</p>
            <br>
            <h4>How it works:</h4>
            <ul>
                <li>👤 Your face is registered during enrollment</li>
                <li>📸 Capture your face using the camera</li>
                <li>🤖 AI recognizes your face</li>
                <li>✅ Attendance is automatically marked</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =========================
# ADMIN DASHBOARD
# =========================
def admin_dashboard():
    col1, col2, col3 = st.columns([0.8, 2, 0.8])
    
    with col2:
        st.markdown("""
        <div class="glass-container" style="text-align: center;">
            <div class="title">👁️ VisionTrack</div>
            <div class="subtitle">Admin Dashboard</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚪 Logout", key="admin_logout"):
        st.session_state.logged_in = False
        st.session_state.user_role = None
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📝 Register Employee", "📋 Attendance Records", "👥 Manage Employees"])
    
    # ===== DASHBOARD TAB =====
    with tab1:
        st.markdown("""
        <div class="glass-card">
            <h2>📊 Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        attendance = load_attendance()
        users = load_users()
        
        total_users = len(users)
        today = datetime.now().strftime("%Y-%m-%d")
        today_attendance = [a for a in attendance if a["date"] == today]
        present = len(today_attendance)
        absent = total_users - present
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 28px;">👥</div>
                <div class="metric-value">{total_users}</div>
                <div class="metric-label">Total Employees</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 28px;">✅</div>
                <div class="metric-value">{present}</div>
                <div class="metric-label">Present Today</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 28px;">❌</div>
                <div class="metric-value">{absent}</div>
                <div class="metric-label">Absent Today</div>
            </div>
            """, unsafe_allow_html=True)
        
        
        
        st.markdown("""
        <div class="glass-card">
            <h3>📅 Today's Attendance</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if len(today_attendance) == 0:
            st.info("ℹ️ No attendance records for today yet")
        else:
            attendance_df = []
            for record in today_attendance:
                attendance_df.append({
                    "👤 Name": record["name"],
                    "⏰ Time": record["time"],
                    "📅 Date": record["date"]
                })
            st.dataframe(attendance_df, use_container_width=True, hide_index=True)
            st.markdown("""
        <div class="glass-card">
            <h3>📅 Today's Attendance Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Check who is present today based on user_id
        present_users_today = {a["user_id"]: a["time"] for a in attendance if a["date"] == today}
        
        attendance_df = []
        
        # Loop through ALL registered users
        for uid, info in users.items():
            if uid in present_users_today:
                status = "✅ Present"
                time_marked = present_users_today[uid]
            else:
                status = "❌ Absent"
                time_marked = "-"
                
            attendance_df.append({
                "👤 Name": info["name"],
                "📊 Status": status,
                "⏰ Time": time_marked
            })
            
        if len(attendance_df) == 0:
            st.info("ℹ️ No employees registered yet.")
        else:
            st.dataframe(attendance_df, use_container_width=True, hide_index=True)
    
    # ===== REGISTER EMPLOYEE TAB =====
    with tab2:
        st.markdown("""
        <div class="glass-card">
            <h2>📝 Register New Employee</h2>
        </div>
        """, unsafe_allow_html=True)
        
        employee_name = st.text_input("👤 Employee Name", key="admin_reg_name", placeholder="Enter full name")
        employee_email = st.text_input("📧 Employee Email", key="admin_reg_email", placeholder="Enter email")
        employee_password = st.text_input("🔑 Set Password", type="password", key="admin_reg_pass", placeholder="Set initial password")
        
        picture = st.camera_input("📸 Capture Face", key=f"admin_cam_{st.session_state.cam_key}")
        
        if st.button("✅ Register Employee", key="register_btn"):
            if employee_name == "":
                st.error("❌ Please enter employee name")
            elif employee_password == "":
                st.error("❌ Please set a password for the employee")
            elif picture is None:
                st.error("❌ Please capture an image")
            else:
                users = load_users()
                
                if any(u["name"].lower() == employee_name.lower() for u in users.values()):
                    st.error("❌ A user with this name is already registered!")
                else:
                    image = Image.open(picture)
                    image = np.array(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                    
                    if len(faces) == 0:
                        st.error("❌ No face detected in the image")
                    else:
                        x, y, w, h = faces[0]
                        face_crop = image[y:y+h, x:x+w]
                        embedding = extract_embedding(face_crop)
                        
                        if embedding is None:
                            st.error("❌ Failed to extract face features")
                        else:
                            user_id = str(uuid.uuid4())
                            embeddings = load_embeddings()
                            
                            users[user_id] = {
                                "name": employee_name,
                                "email": employee_email,
                                "password": employee_password, 
                                "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            embeddings[user_id] = embedding
                            
                            save_users(users)
                            save_embeddings(embeddings)
                            
                            st.success(f"✅ {employee_name} registered successfully!")
                            st.session_state.cam_key += 1 
                            st.rerun()
    
    # ===== ADMIN ATTENDANCE RECORDS =====
    with tab3:
        attendance = load_attendance()
        users = load_users()
        
        col1, col2 = st.columns(2)
        with col1:
            filter_date = st.date_input("📅 Select Date", value=datetime.now())
        with col2:
            filter_name = st.selectbox("👤 Select Employee", ["All"] + [u["name"] for u in users.values()])
        
        filter_date_str = filter_date.strftime("%Y-%m-%d")
        today = datetime.now().date()

        if filter_name != "All":
            # LOGIC FOR INDIVIDUAL EMPLOYEE STATS
            selected_uid = next((uid for uid, info in users.items() if info["name"] == filter_name), None)
            info = users[selected_uid]
            reg_date = datetime.strptime(info["registered_at"].split(" ")[0], "%Y-%m-%d").date()
            
            # Calculate Monthly Absences for Warning
            all_days = [reg_date + timedelta(days=x) for x in range((today - reg_date).days + 1)]
            current_month = today.strftime("%Y-%m")
            month_days = [d for d in all_days if d.strftime("%Y-%m") == current_month]
            month_present = {a["date"] for a in attendance if a["user_id"] == selected_uid and a["date"].startswith(current_month)}
            month_absent_count = len(month_days) - len(month_present)

            if month_absent_count > 3:
                st.warning(f"❗ **Admin Note:** {filter_name} has exceeded 3 absences this month ({month_absent_count} days).")

            # Calculate Totals for Graph
            total_present = len([a for a in attendance if a["user_id"] == selected_uid])
            total_absent = len(all_days) - total_present
            
            st.markdown(f"### 📊 {filter_name}'s Lifetime Attendance")
            
            # Use columns and height to make the graph smaller
            graph_col, empty_col = st.columns([1, 2]) # 1 part graph, 2 parts empty space
            with graph_col:
                st.bar_chart({"Status": ["Present", "Absent"], "Days": [total_present, total_absent]}, x="Status", y="Days", height=200)

        # RENDER THE DATA TABLE
        attendance_df = []
        for uid, info in users.items():
            if filter_name == "All" or info["name"] == filter_name:
                reg_date = datetime.strptime(info["registered_at"].split(" ")[0], "%Y-%m-%d").date()
                
                if filter_date < reg_date:
                    status, time_marked = "➖ Not Registered", "-"
                else:
                    record = next((a for a in attendance if a["user_id"] == uid and a["date"] == filter_date_str), None)
                    status = "✅ Present" if record else "❌ Absent"
                    time_marked = record["time"] if record else "-"
                
                attendance_df.append({
                    "👤 Name": info["name"],
                    "📊 Status": status,
                    "⏰ Time": time_marked,
                    "📅 Registered Date": info["registered_at"]
                })
        
        if not attendance_df:
            st.info("ℹ️ No records found.")
        else:
            st.dataframe(attendance_df, use_container_width=True, hide_index=True)
    
    # ===== MANAGE EMPLOYEES TAB =====
    with tab4:
        st.markdown("""
        <div class="glass-card">
            <h2>👥 Manage Employees</h2>
            <p>View or Remove registered employees.</p>
        </div>
        """, unsafe_allow_html=True)
        
        users = load_users()
        
        if len(users) == 0:
            st.info("ℹ️ No employees registered yet")
        else:
            for uid, info in list(users.items()):
                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 3, 3, 1.5])
                    with c1:
                        st.markdown(f"**{info['name']}**")
                    with c2:
                        st.markdown(f"📧 {info.get('email', 'N/A')}")
                    with c3:
                        st.markdown(f"📅 {info.get('registered_at', 'N/A')}")
                    with c4:
                        if st.button("🗑️ Remove", key=f"del_{uid}"):
                            del users[uid]
                            
                            embeddings = load_embeddings()
                            if uid in embeddings:
                                del embeddings[uid]
                                
                            attendance = load_attendance()
                            updated_attendance = [record for record in attendance if record.get("user_id") != uid]
                            
                            save_users(users)
                            save_embeddings(embeddings)
                            save_attendance(updated_attendance)
                            
                            st.success(f"Removed {info['name']} and their attendance records")
                            st.rerun()
                st.markdown("<hr style='margin:0.5em 0;'>", unsafe_allow_html=True)

# =========================
# MAIN APP
# =========================
if not st.session_state.logged_in:
    login_page()
elif st.session_state.user_role == "employee":
    employee_dashboard()
elif st.session_state.user_role == "admin":
    admin_dashboard()
