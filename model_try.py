import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_offensive_model():
    # Load your dataset
    df = pd.read_csv("data_r3.csv")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['Offensive'], test_size=0.2, random_state=42)

    # Build a pipeline with TF-IDF and a simple classifier (e.g., SVM)
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', SVC(kernel='linear'))
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    return model_pipeline

# Function to check offensive content


def check_offensive_content(text_representation, offensive_model):
    # Use the offensive content detection model
    prediction = offensive_model.predict([text_representation])

    # Interpretation of the prediction
    if prediction[0] == 1:
        return "This content is considered offensive."
    else:
        return "This content is not offensive."


s = "I got a black man painting himself white "
offensive_model = train_offensive_model()

# Check for offensive content
result = check_offensive_content(s, offensive_model)

print(result)


print("Test")
count = 0
df_test = ["Black man turing into a white man for interview",
           "Women are made to clean", "dad beats his son because he is gay"]
for i in df_test:
    print(i)
    offensive_model = train_offensive_model()
    result = check_offensive_content(i, offensive_model)
    if result == "This content is considered offensive.":
        count += 1
    print(result)
print(count, "/ 3")
