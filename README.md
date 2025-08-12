# Email Classification Using Deep Learning (Spam • Scam • Safe)

Classifies emails into **Spam**, **Scam (phishing)**, and **Safe** using:

- **LSTM (GloVe embeddings)**
- **BERT (bert‑base‑uncased)**
- **Hypertuned BERT (Optuna search)**
- **Hybrid BERT + BiLSTM**

Runs end‑to‑end on **Google Colab**.  
Start with `preprocessing.py` to create a single CSV used by all models.


## 📂 Repository Structure

```plaintext
├── data/                                       # Contains datasets
│   ├── phishingemails.csv                      # Kaggle "Phishing Email Detection"
│   ├── spam-mails-dataset.csv                  # Kaggle "Spam Mails Dataset"
│   └── combined_emails.csv                     # Created by preprocessing.py
├── bert.py                                     # Fine-tune BERT (bert-base-uncased)
├── bert_lstm.py                                # Hybrid: BERT embeddings + BiLSTM
├── Deep_Learning_Project_Emails_final_report.pdf  # Final project report
├── Email Classification Using Deep Learning...pptx # Presentation slides
├── hypertuned_bert.py                          # Train BERT with best hyperparameters
├── hypertuning_bert_parameter_search.py        # Optuna search for BERT hyperparameters
├── lstm.py                                     # LSTM model with GloVe embeddings
├── preprocessing.py                            # Preprocess datasets & create combined CSV
├── readme                                      # Legacy readme text file
```

## 🚀 Quick Start (Google Colab)

1. Open **Google Colab → New Notebook**.
2. Put all project files (and the `data/` folder) in your Colab working directory  
   (either upload them or mount Drive and `cd` into the folder).

```python
# (Optional) Mount Drive and cd into your project
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/<your-path-to-project>

# Install dependencies
!pip install -q transformers==4.* torch==2.* scikit-learn pandas numpy nltk optuna matplotlib

# NLTK resources for preprocessing
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```
### 📄 Running the Project
```python
# 1. Generate the unified CSV (must be done once)
!python preprocessing.py   # writes ./data/combined_emails.csv

# 2. Run any model script (one at a time)

# LSTM baseline
!python lstm.py

# BERT baseline (fine-tuning)
!python bert.py

# Hyperparameter search for BERT (prints best params)
!python hypertuning_bert_parameter_search.py

# Train BERT with best params found above
!python hypertuned_bert.py

# Hybrid: BERT embeddings → BiLSTM classifier
!python bert_lstm.py

# All scripts print metrics to the console and save figures/artifacts into ./results/
# (and may save trained models under a model-specific folder).
```

### 🧹 What `preprocessing.py` Does
- Loads the two Kaggle datasets from `data/`
- Harmonizes labels to `Safe Email`, `Spam`, `Scam`
- Balances classes (downsampling)
- Cleans text: strip HTML, lowercase, remove special chars/stopwords, lemmatize
- **Output:** `data/combined_emails.csv` used by every model script

### 🧠 Models & Scripts

**lstm.py**
- Tokenization + padding; loads **GloVe** embeddings
- 2× LSTM with dropout, early stopping, checkpoints
- Saves confusion matrix and metrics table in `results/`

**bert.py**
- Fine-tunes **bert-base-uncased** for 3-class classification
- Uses HF `Trainer`; saves metrics, confusion matrix, and model artefacts

**hypertuning_bert_parameter_search.py**
- **Optuna** search over LR, batch size, epochs, weight decay (F1 objective)
- Prints/saves the best hyperparameters

**hypertuned_bert.py**
- Trains BERT with the best hyperparameters from the search

**bert_lstm.py**
- Uses BERT hidden states → **BiLSTM** → dropout → dense softmax
- Aims for lowest validation loss / strongest robustness

### 📊 Typical Results (on our split)
- LSTM: ~0.68 accuracy  
- BERT baseline: ~0.97 accuracy (val loss ≈ 0.113)  
- Hypertuned BERT: accuracy similar, improved val loss ≈ 0.098  
- BERT + BiLSTM: best val loss ≈ 0.0885  

> Exact numbers depend on your split, seed, and compute.

### 🔮 Using Trained Models for Inference
**BERT family (baseline / hypertuned / hybrid):**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_dir = "./saved_models/bert_best"   # change to your save path
tok = AutoTokenizer.from_pretrained(model_dir)
mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)

labels = ["Safe Email","Spam","Scam"]    # keep order consistent with training

def predict(texts):
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = mdl(**enc).logits
    preds = torch.argmax(logits, dim=-1).tolist()
    return [labels[i] for i in preds]

print(predict(["⚠️ You won a prize, click here!", 
               "Meeting moved to 3pm, see calendar"]))
```
**LSTM:** load the saved `.h5` (or checkpoint) and reuse the exact tokenizer fitted during training.

### 🧪 Reproducing Figures
- Confusion matrices and metric tables for each model are saved in `results/`  
  (e.g., `lstmconf.png`, `bertconf.png`, `hybertconf.png`, `lstmres.png`, `bertres.png`, `hybertres.png`).
- Dataset composition plot is saved as `datasetcomposition.png`.
- Hybrid model training curves may appear as `bertlstmloss.png`, etc.

### ⚙️ Tips & Troubleshooting
- **`data/combined_emails.csv not found`** → Run `preprocessing.py` and confirm paths.
- **CUDA OOM** → Use a smaller `max_length`, reduce batch size, or switch runtime to CPU.
- **Tokenizer mismatch** → Always load tokenizer and model from the same save directory.
- **Different file names** → Update paths at the top of each script.

### 📜 License & Acknowledgments
- Academic use.
- Datasets from Kaggle; models via Hugging Face Transformers.
- Thanks to Professor Chandan Reddy for guidance.

### 📌 Workflow Recap
1. Put raw CSVs in `data/`
2. Run `preprocessing.py`
3. Run any of:
   - `lstm.py`
   - `bert.py`
   - `hypertuning_bert_parameter_search.py` → `hypertuned_bert.py`
   - `bert_lstm.py`
4. Check `results/` for figures and saved metrics

