#Please read the readme file before running the code

import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, TFBertForSequenceClassification, BertForSequenceClassification, Trainer, TrainingArguments, BertModel
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
from torch import nn


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


# Define a custom model with LSTM on top of BERT
class BertLSTMClassifier(nn.Module):
    def __init__(self, num_labels, lstm_hidden_size=256, lstm_layers=1, dropout=0.3):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden_size, num_labels)  # 2 for bidirectional

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass inputs through BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Get sequence output
        sequence_output = bert_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        # Pass through LSTM
        lstm_output, _ = self.lstm(sequence_output)  # Shape: (batch_size, seq_len, 2 * hidden_size)
        # Use the output of the [CLS] token (first token) from the LSTM
        cls_output = lstm_output[:, 0, :]  # Shape: (batch_size, 2 * hidden_size)
        # Apply dropout
        cls_output = self.dropout(cls_output)
        # Pass through the classifier
        logits = self.fc(cls_output)  # Shape: (batch_size, num_labels)
        return logits

# Update your dataset class if needed
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

# Prepare the datasets
train_dataset = EmailDataset(X_train, y_train)
test_dataset = EmailDataset(X_test, y_test)

# Initialize the model with LSTM
num_labels = 3  # Replace with the actual number of classes
model = BertLSTMClassifier(num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",  # Updated from evaluation_strategy
    learning_rate=1.5451037291687613e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=2.4486870650141793e-06,
    logging_dir='./logs',
)

# Define a custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, labels)
        return (loss, outputs) if return_outputs else loss


# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train and Evaluate
trainer.train()
trainer.evaluate()



# Create a zip file of the saved model
shutil.make_archive("/content/bert_lstm_model", 'zip', "./bert_lstm_model")


# Download the zip file to your local computer
files.download("/content/bert_lstm_model.zip")