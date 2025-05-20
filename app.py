import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import bcrypt
import os

st.set_page_config(page_title="SmartHouse AI", layout="centered")

# ----------------------
# User Auth Setup
# ----------------------
USER_FILE = "users_secure.csv"
if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_FILE, index=False)

def hash_pass(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_pass(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def get_users():
    return pd.read_csv(USER_FILE)

def add_user(username, password):
    df = get_users()
    hashed = hash_pass(password)
    df = pd.concat([df, pd.DataFrame([[username, hashed]], columns=["username", "password"])])
    df.to_csv(USER_FILE, index=False)

# ----------------------
# Session Setup
# ----------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""

def login():
    st.subheader("🔐 Login to SmartHouse AI")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = get_users()
        if username in users["username"].values:
            stored_hash = users[users["username"] == username]["password"].values[0]
            if check_pass(password, stored_hash):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("✅ Login Successful")
            else:
                st.error("❌ Incorrect Password")
        else:
            st.error("⚠️ Username not found")

def signup():
    st.subheader("📝 Create a SmartHouse AI Account")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    if st.button("Create Account"):
        users = get_users()
        if username in users["username"].values:
            st.warning("⚠️ Username already taken.")
        else:
            add_user(username, password)
            st.success("✅ Account created! You can login now.")

auth_action = st.sidebar.radio("User Access", ["Login", "Signup"])
if not st.session_state.authenticated:
    if auth_action == "Login":
        login()
    else:
        signup()
    st.stop()

# ----------------------
# Training Data
# ----------------------
train_data = {
    'area': [1000, 1500, 1800, 2400, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'price': [50, 65, 70, 90, 110]
}
df = pd.DataFrame(train_data)
X = df[['area', 'bedrooms']]
y = df['price']
model = LinearRegression()
model.fit(X, y)

# ----------------------
# Country & State Multipliers
# ----------------------
state_price_factor = {
    "Maharashtra": 1.25, "Delhi": 1.30, "Karnataka": 1.10,
    "Tamil Nadu": 1.05, "West Bengal": 0.95, "Uttar Pradesh": 0.90,
    "Bihar": 0.80, "Gujarat": 1.00, "Rajasthan": 0.85, "Kerala": 1.15
}

# ----------------------
# UI
# ----------------------
st.title("🏡 SmartHouse AI - 2025")
st.markdown(f"Welcome, **{st.session_state.username}** 👋")

country = st.selectbox("🌐 Country", ["India"])
state = st.selectbox("📍 State", list(state_price_factor.keys()))
area = st.slider("📏 Area (sqft)", 500, 5000, 1500, step=100)
bedrooms = st.selectbox("🛏️ Bedrooms", [1, 2, 3, 4, 5])

def predict(area, bedrooms, factor):
    base_price = model.predict([[area, bedrooms]])[0]
    return base_price * factor

if st.button("🚀 Predict"):
    factor = state_price_factor[state]
    predicted_price = predict(area, bedrooms, factor)
    st.success(f"Estimated Price in {state}: ₹{predicted_price:.2f} Lakh")

    # Plot
    st.subheader("📊 Price Visualization")
    fig, ax = plt.subplots()
    ax.scatter(df['area'], df['price'], label="Training Data")
    ax.scatter(area, predicted_price / factor, color='red', marker='x', s=100, label="Your Input")
    ax.set_xlabel("Area (sqft)")
    ax.set_ylabel("Price (Lakh ₹)")
    ax.legend()
    st.pyplot(fig)

    # Metrics
    r2 = r2_score(y, model.predict(X))
    mae = mean_absolute_error(y, model.predict(X))
    st.markdown("### 📈 Model Performance")
    st.metric("R² Score", f"{r2:.2f}")
    st.metric("MAE", f"{mae:.2f} Lakh ₹")

    # Smart AI Tips
    st.markdown("### 🤖 AI Suggestions")
    tips = []
    if area < 1000:
        tips.append("📐 Small area — consider expanding or choosing efficient layout.")
    if bedrooms < 3:
        tips.append("🛏️ Additional bedroom may increase resale value.")
    if factor < 1.0:
        tips.append("📍 Great for low-cost investment in developing state.")
    else:
        tips.append("🏙️ You're in a high-demand region — invest in quality interiors.")
    for t in tips:
        st.markdown(f"- {t}")

# ----------------------
# Dataset Display
# ----------------------
with st.expander("📂 Show Training Dataset"):
    st.dataframe(df)

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("🔐 **Secure. Smart. Scalable.** | Built with ❤️ using Streamlit & AI (2025)")
