import pickle

# Step 1: Load model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Step 2: Transform input text using the SAME vectorizer
new_text = "Congratulations! You won a free ticket."
vector_input = vectorizer.transform([new_text])  # No fit, only transform

# Step 3: Predict
result = model.predict(vector_input)[0]
print(f"📢 Prediction: {result}")
