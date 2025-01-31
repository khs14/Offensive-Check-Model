import streamlit as st
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
df = pd.read_csv("data_r3.csv")

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Offensive'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# Tokenize text data
max_words = 10000  # Adjust based on your vocabulary size
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
max_sequence_length = 100  # Adjust based on your sequence length
X_train_padded = pad_sequences(
    X_train_seq, maxlen=max_sequence_length, padding='post')
X_test_padded = pad_sequences(
    X_test_seq, maxlen=max_sequence_length, padding='post')

# Build CNN model
embedding_dim = 50  # Adjust based on your embedding dimension
filters = 64
kernel_size = 3

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim,
          input_length=max_sequence_length))
model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 7  # Adjust based on your training preferences
batch_size = 32
model.fit(X_train_padded, y_train, epochs=epochs,
          batch_size=batch_size, validation_data=(X_test_padded, y_test))

# Evaluate the model
# loss, accuracy = model.evaluate(X_test_padded, y_test)
# print(f"Test Accuracy: {accuracy}")

# Assuming you have already trained the CNN model using the provided code

# Function to preprocess and tokenize text


def preprocess_text(text):
    # Tokenize text
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(
        text_seq, maxlen=max_sequence_length, padding='post')
    return text_padded


def main():
    st.title("Offensive Marketing Campaign Check")

    # User input
    text_input = st.text_area("Enter text:")


    # Check Offense button
    if st.button("Check Offense"):
        # Display progress bar for 5 seconds
        with st.spinner("Checking Offense..."):
            time.sleep(1)
            st.success("Done!")

            # Perform offensive content detection
            result = detect_offensive_content(text_input, uploaded_file)

            # Display the result
            st.write(f"**Result:** {result}")


def detect_offensive_content(text_input):
    result = perform_offensive_content_detection(text_input)

    return result




def perform_offensive_content_detection(text):
    test_text_padded = preprocess_text(text)
    prediction = model.predict(test_text_padded)
    if prediction[0, 0] >= 0.5:
        result = "This content is considered offensive."
    else:
        result = "This content is not offensive."

    return result


if __name__ == "__main__":
    main()
