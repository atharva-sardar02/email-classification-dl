import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Bidirectional
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load datasets
spam_df = pd.read_csv('spam-mails-dataset.csv')  # Adjust filename as needed
phishing_df = pd.read_csv('phishingemails.csv')  # Adjust filename as needed

# Map labels to Spam and Scam
spam_df['label'] = spam_df['label'].map({'spam': 'Spam'})  # Only keep "Spam"
phishing_df['Email Type'] = phishing_df['Email Type'].map({'Phishing Email': 'Scam'})  # Only keep "Scam"

# Remove 'Safe Email' rows
spam_df = spam_df[spam_df['label'] == 'Spam']
phishing_df = phishing_df[phishing_df['Email Type'] == 'Scam']

# Rename columns for consistency
spam_df = spam_df.rename(columns={'text': 'Email Text', 'label': 'Email Type'})
phishing_df = phishing_df[['Email Text', 'Email Type']]

# Combine both datasets
combined_df = pd.concat([spam_df[['Email Text', 'Email Type']], phishing_df], ignore_index=True)

# Just use 100 samples for now
combined_df = combined_df.sample(40, random_state=1000)

# Encode labels as integers (Spam = 0, Scam = 1)
label_encoder = LabelEncoder()
combined_df['label_num'] = label_encoder.fit_transform(combined_df['Email Type'])

# Extract the target and input
X = combined_df['Email Text']
X = X.fillna('').astype(str)
y = combined_df['label_num']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Resample the training data using SMOTE to handle class imbalance (after TF-IDF conversion)
# Note: SMOTE works on numerical features like TF-IDF, not raw text. However, BERT doesn't need SMOTE,
# so we will just use BERT tokenization here and train on the whole data.
# We'll show how SMOTE can be used with TF-IDF if you still want to try TF-IDF.

# Load the BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the inputs
def tokenize_bert(sentences, tokenizer, max_length=100):
    return tokenizer(
        list(sentences),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# Tokenize training and testing data using BERT tokenizer
train_encodings = tokenize_bert(X_train, bert_tokenizer)
test_encodings = tokenize_bert(X_test, bert_tokenizer)

# Define the BERT model for 2 categories (Spam and Scam)
model_bert = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Calculate class weights to address the imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Compile the BERT model
model_bert.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Train the model with class weights
model_bert.fit(
    {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},  # Tokenized inputs
    y_train,
    validation_data=(
        {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']},  # Validation tokenized inputs
        y_test
    ),
    epochs=3,
    batch_size=16,
    class_weight=class_weight_dict  # Add class weights here
)

# Evaluate the BERT model
bert_preds = model_bert.predict(
    {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']}
).logits
bert_preds = np.argmax(bert_preds, axis=1)

# Print classification report for Spam and Scam classes
print("Classification Report:\n")
print(classification_report(y_test, bert_preds, target_names=label_encoder.classes_))

# Calculate ROC-AUC for better evaluation of the model's performance on both classes
y_pred_proba = model_bert.predict(
    {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']}
).logits
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])  # AUC for the 'Scam' class
print(f"\nROC AUC Score: {roc_auc}")

# Save the model
# model_bert.save_pretrained('bert_model')

# # Load the model (if needed)
# loaded_model = TFBertForSequenceClassification.from_pretrained('bert_model')
