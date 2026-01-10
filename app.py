import streamlit as st
import pickle

# 1. Load the trained model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# 2. Set up the Streamlit page
st.set_page_config(page_title="Email Spam Classifier", page_icon="📩")

st.title("📩 AI Email Classifier")
st.write("Enter an email or SMS message below to check if it's Spam or Safe.")

# 3. Input area
user_input = st.text_area("Paste message here:", height=150)

# 4. Prediction logic
if st.button("Classify Message"):
    if user_input:
        # Preprocess the input like we did in training
        data = [user_input]
        vectorized_input = vectorizer.transform(data)
        
        # Predict
        prediction = model.predict(vectorized_input)
        
        # Display Result
        if prediction[0] == 1:
            st.error("🚨 SPAM DETECTED!")
            st.write("This message looks suspicious.")
        else:
            st.success("✅ NOT SPAM")
            st.write("This message looks safe.")
    else:
        st.warning("Please enter a message first.")

# Footer
st.markdown("---")
st.caption("Built with Python & Streamlit by Abdullah Ahsan")
