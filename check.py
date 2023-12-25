import pandas as pd
df = pd.read_csv("data_r3.csv")
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

