import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load your dataset
df = pd.read_csv("data_r3.csv")

# Function to extract linguistic features using spaCy


def extract_features(text):
    doc = nlp(text)

    # Example: Extracting the average word length
    avg_word_length = sum(len(token.text) for token in doc) / len(doc)

    # You can extract more features based on your requirements

    return [avg_word_length]


# Add spaCy features to the dataset
df['spacy_features'] = df['text'].apply(extract_features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['spacy_features'].tolist(), df['Offensive'], test_size=0.2, random_state=42)

# Build a pipeline with a simple classifier (e.g., SVM)
model_pipeline = Pipeline([
    ('classifier', SVC(kernel='linear'))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


# Assuming you have already trained the model using the provided code

# Test case
test_text = "For ten days, we can't use the 'N' word."

# Extract spaCy features for the test case
test_features = extract_features(test_text)

# Make a prediction using the trained model
prediction = model_pipeline.predict([test_features])

# Interpretation of the prediction
if prediction[0] == 1:
    result = "This content is considered offensive."
else:
    result = "This content is not offensive."

# Display the result for the test case
print(f"Test Case: {test_text}")
print(f"Result: {result}")
