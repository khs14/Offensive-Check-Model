import pandas as pd
import re
from collections import Counter
import pandas as pd
df = pd.read_csv("data_r1.csv")
df = df.dropna(how='any')
print(len(df))
df.info()
df.to_csv('data_r2.csv')

df3 = pd.read_csv("data_r2.csv")
df3 = df3[["class", "text", "Offensive"]]
df3.to_csv("data_r3.csv")
print(len(df3))
