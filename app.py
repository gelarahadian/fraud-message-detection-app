import streamlit as st
import pickle

import streamlit as st
import pickle

# Load model
with open('spam_detector.pkl', 'rb') as f:
    model = pickle.load(f)

# UI
st.title("📩 Fraud Message Detection")
st.write("Detect whether a message is spam/fraud or not.")

# User input
user_input = st.text_area("Enter your message below:")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Message cannot be empty.")
    else:
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]

        if prediction == 1:
            st.error("⚠️ This message is detected as **SPAM/FRAUD**.")
        else:
            st.success("✅ This message is **safe** (not spam).")

        st.write(f"Spam probability: `{proba[1]:.2f}` | Safe: `{proba[0]:.2f}`")
