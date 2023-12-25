import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Load your dataset

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
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test Accuracy: {accuracy}")

# Assuming you have already trained the CNN model using the provided code

# Function to preprocess and tokenize text


def preprocess_text(text):
    # Tokenize text
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(
        text_seq, maxlen=max_sequence_length, padding='post')
    return text_padded


print("Test")
count = 0
df_test = ["Black man turing into a white man for interview",
           "Women are made to clean", "dad beats his son because he is gay"]
for i in df_test:
    print(i)
    test_text_padded = preprocess_text(i)
    prediction = model.predict(test_text_padded)
    if prediction[0, 0] >= 0.5:
        result = "This content is considered offensive."
        count += 1
    else:
        result = "This content is not offensive."


print(count, "/3")
