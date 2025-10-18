# Financial Causal Relation Detector

### _Intelligent Financial Text Analysis using Deep Learning and NLP_

This repository provides an **advanced Financial Causal Relation Detection System** that identifies **cause–effect relationships** in financial text data.  
It leverages a **hybrid deep learning architecture** combining **Bidirectional LSTM**, **multi-head attention**, and **contextual encoding layers** for robust causal inference.

A **Flask web application** is included for **real-time predictions** and **interactive analysis** of user-input financial sentences.

---

## Features

- Automatic detection of cause–effect relationships in financial texts
- Domain-specific preprocessing for financial entities (revenues, profits, interest rates, etc.)
- Hybrid deep learning model leveraging **LSTM** and **attention mechanisms**
- Real-time web interface built with **Flask**

---

## Project Structure

```text
financial-causal-relation-detector/
├── Model_Training_Code.ipynb       # Model architecture and training notebook
├── WebApp/                         # Flask web application
│   ├── app.py                      # Flask backend
│   ├── static/                     # Frontend assets (CSS, JS, logos)
│   ├── templates/                  # HTML templates for Flask app
|   ├── requirements.txt             # Dependencies
│   └── models                      # models
├── updated_final_dataset.csv       # Input dataset for training
└── README.md                       # Project documentation
```

---

## Model Overview

### **1. Data Processing**

- **Tokenization & Numericalization:** Custom tokenizer to preserve numbers and punctuation
- **Vocabulary Building:** Frequency-based word indexing
- **Dataset Class:** `FinancialCausalDataset` prepares span masks for cause/effect detection

### **2. Model Architecture**

The **FinancialCausalDetector** model includes:

- **Embedding Layer:** Token-to-vector transformation
- **Bidirectional LSTM:** Captures context from both directions
- **Causal Context Layer:** Multi-head self-attention for token-level causal relevance
- **Feed-forward Layers:** Contextual refinement
- **Output Heads:** Predict cause/effect spans and relation type

  - `Cause-Effect`, `Effect-Cause`, or `None`

### **3. Loss & Training**

- Loss: `BCEWithLogitsLoss` for span prediction
- Optimizer: `AdamW`
- Automatic model checkpointing on best validation loss

---

## Flask Web Application

The Flask app enables **real-time inference** on financial text.

| Route      | Method | Description                                                |
| ---------- | ------ | ---------------------------------------------------------- |
| `/`        | GET    | Render the input page                                      |
| `/analyze` | POST   | Process user text input and return detected causes/effects |

### Example Run

```bash
python app.py
```

Open your browser at:
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## Usage

### 1️ Clone the Repository

```bash
git clone https://github.com/your-username/financial-causal-relation-detector.git
cd financial-causal-relation-detector
```

### 2️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️ Train the Model

Edit the dataset path and run:

```bash
python Model_Training_Code.ipynb
```

Or within a notebook:

```python
model, vocab = main(csv_path='updated_final_dataset.csv', savedir='financialmodel')
```

### 4️ Run Flask Web App

```bash
python app.py
```

---

## Example Predictions

**Input Sentence:**

> “Due to the increase in interest rates, mortgage applications decreased by 10% last month.”

**Output:**

```
Detected cause: increase in interest rates
Detected effect: mortgage applications decreased
Relation type: Cause-Effect
Confidence: Cause (0.91), Effect (0.89)
```

---

## Evaluation

- Evaluated on **20% test split**
- Metrics: **F1-score**, **span-level accuracy**, and **relation classification**
- Manual verification confirms strong generalization to unseen financial text

---

## Technical Stack

| Category       | Technologies                        |
| -------------- | ----------------------------------- |
| **Language**   | Python 3.11                         |
| **Frameworks** | PyTorch, Flask                      |
| **Libraries**  | scikit-learn, pandas, numpy, pickle |
| **Frontend**   | HTML, CSS, JavaScript               |
| **Tools**      | Jupyter, Git, VS Code               |

---

## Potential Extensions

- Integration with **live financial news APIs**
- Transformer-based models (e.g., **FinBERT**, **RoBERTa**)
- Graph visualization for causal chains
- Multi-lingual corpus support

---
