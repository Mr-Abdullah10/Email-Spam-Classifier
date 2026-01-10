import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# 1. Load Data
# Note: The encoding might differ, 'latin-1' usually fixes CSV issues
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: spam.csv not found. Please make sure the file is in the same folder.")
    exit()

# 2. Preprocessing
# We only need the 'v1' (label) and 'v2' (text) columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers: spam=1, ham=0
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label_num']

# 3. Feature Extraction (Turning text into numbers)
cv = CountVectorizer()
X_vectorized = cv.fit_transform(X)

# 4. Split Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 5. Train Model (Naive Bayes is great for text)
model = MultinomialNB()
model.fit(X_train, y_train)

print(f"Model Training Complete!")
print(f"Accuracy Score: {model.score(X_test, y_test) * 100:.2f}%")

# 6. Save the Model and Vectorizer
# We need both to run the app later
pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

print("Model and Vectorizer saved successfully.")
