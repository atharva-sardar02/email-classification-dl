#Please read the readme file before running the code

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Bidirectional
from sklearn.utils import resample
import re

from google.colab import drive
drive.mount("/content/drive")

# Load datasets
dataset_path_1 = '/content/drive/My Drive/Colab Notebooks/spam-mails-dataset.csv'
dataset_path_2 = '/content/drive/My Drive/Colab Notebooks/phishingemails.csv'
spam_df = pd.read_csv(dataset_path_1)
phishing_df = pd.read_csv(dataset_path_2)
print(spam_df.head())
print(phishing_df.head())

# Map labels to common categories
spam_df['label'] = spam_df['label'].map({'spam': 'Spam', 'ham': 'Safe Email'})
phishing_df['Email Type'] = phishing_df['Email Type'].map({'Phishing Email': 'Scam', 'Safe Email': 'Safe Email'})

# Rename columns for consistency
spam_df = spam_df.rename(columns={'text': 'Email Text', 'label': 'Email Type'})
phishing_df = phishing_df[['Email Text', 'Email Type']]

# Combine both datasets
combined_df = pd.concat([spam_df[['Email Text', 'Email Type']], phishing_df], ignore_index=True)

df=combined_df
print(df.head())

df.groupby('Email Type').describe()

df=df.dropna()

df_safe = df[df['Email Type'] == 'Safe Email'].drop_duplicates(subset='Email Text')
df_scam = df[df['Email Type'] == 'Scam'].drop_duplicates(subset='Email Text')
df_spam = df[df['Email Type'] == 'Spam'].drop_duplicates(subset='Email Text')

print(f"Unique Safe Emails: {len(df_safe)}")
print(f"Unique Scam Emails: {len(df_scam)}")
print(f"Unique Spam Emails: {len(df_spam)}")

min_class_size = min(len(df_safe), len(df_scam), len(df_spam))

df_safe_balanced = resample(df_safe, n_samples=min_class_size, random_state=1000)
df_scam_balanced = resample(df_scam, n_samples=min_class_size, random_state=1000)
df_spam_balanced = resample(df_spam, n_samples=min_class_size, random_state=1000)

df_balanced = pd.concat([df_safe_balanced, df_scam_balanced, df_spam_balanced])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced Dataset Size: {len(df_balanced)}")

df_balanced.head()

df_balanced.groupby('Email Type').describe()

df_balanced.to_csv('preprocessed_email_data.csv', index=False)

from google.colab import files

# This will download the CSV file to your local system
files.download('preprocessed_email_data.csv')
