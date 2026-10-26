import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../data/sms-spam-collection.csv")
df.head()
df.isnull().sum()
df.info()
df['label'].value_counts()

df['label'].value_counts() / len(df) * 100
df.duplicated().sum()
df_no_drop = df.drop_duplicates()
df_no_drop['label'].value_counts()

# Message Length Analysis
df['message_length'] = df['message'].apply(lambda x: len(x))
df.head()
spam_avg = df[df['label']== 'spam']['message_length'].mean()
ham_avg = df[df['label']== 'ham']['message_length'].mean()
max_length = df['message_length'].max()
min_length = df['message_length'].min()
print(spam_avg)
print(ham_avg)
print(max_length)
print(min_length)
df[df['message_length']== 910 ] 
df[df['message_length']== 2 ] 

# Message Length Visualization

plt.Figure(figsize=(10,5))
sns.histplot(df['message_length'])
plt.title('Distribution of Message Length')
plt.xlabel('Message Length')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8,5))
sns.boxenplot(df, x='label', y='message_length')
plt.title('message length by class')
plt.xlabel('message type')
plt.ylabel('message length')

plt.show()


