#Please read the readme file before running the code

import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, TFBertForSequenceClassification, Trainer, TrainingArguments, BertForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Bidirectional, Dropout
from sklearn.utils import resample
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from google.colab import drive, files
import shutil
import optuna
from torch.utils.data import Dataset



drive.mount("/content/drive")

dataset_path_1 = '/content/drive/My Drive/Colab Notebooks/preprocessed_email_data.csv'
df = pd.read_csv(dataset_path_1)
print(df.head())

# Clean the email text
def clean_text(text):
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

df['clean_text'] = df['Email Text'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Email Type'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=1000
)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts.iloc[idx],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }

train_dataset = EmailDataset(X_train, y_train)
test_dataset = EmailDataset(X_test, y_test)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=3)

# Define training arguments which we get from the hypertuning of bert
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=1.5451037291687613e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=2.4486870650141793e-06,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train and Evaluate
trainer.train()
trainer.evaluate()

# Evaluate BERT
bert_preds = trainer.predict(test_dataset).predictions.argmax(axis=1)
print("BERT Evaluation:")
print(classification_report(y_test, bert_preds, target_names=label_encoder.classes_))

# Generate the classification report
report = classification_report(y_test, bert_preds, target_names=label_encoder.classes_, output_dict=True)