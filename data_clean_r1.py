import re
from collections import Counter
import pandas as pd

df_og = pd.read_csv("data.csv")

df_og = df_og[["class", "text"]]
df2 = pd.read_csv("offensive_1.csv", encoding='unicode_escape')
df3 = pd.read_csv("non_offensive1.csv")
df_add = pd.concat([df2, df3])
df = pd.concat([df_add, df_og])


def is_greater(value):  # replaces class of 2 with  value 1
    if value <= 1:
        return 1
    else:
        return 0


df["Offensive"] = df["class"].apply(is_greater)


# Count occurrences of a specific value (e.g., 1) in the column
value_counts = df['Offensive'].value_counts()

# Display the count for the specific value
specific_value = 1
count_of_specific_value = value_counts.get(specific_value, 0)
print(f"Count of {specific_value} in the column: {count_of_specific_value}")


# Display the count for the specific value
specific_value = 0
count_of_specific_value = value_counts.get(specific_value, 0)
print(f"Count of {specific_value} in the column: {count_of_specific_value}")

print(len(df))


def remove_bad(word, no_of_reduction):
    count = 0
    reduction = 0
    for index, row in df.iterrows():
        message = row[1].lower()

        if word in message:
            reduction += 1
            if reduction <= no_of_reduction:
                df.drop(index, inplace=True)


def calc_word_count(df, word):
    count_of_specific_word = df['text'].str.contains(word, case=False).sum()
    return count_of_specific_word


worst_words = ["hoe", "hoes", "niggas", "nigger",
               "niggers", "faggot", "bitch", "bitches", "pussy"]

for i in worst_words:
    remove_bad(i, int(0.99*calc_word_count(df, i)))


df.to_csv('data_r1.csv')

print("done")
