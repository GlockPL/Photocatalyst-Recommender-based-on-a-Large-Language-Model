# Photocatalyst Recommender Based on a Large Language Model

A complete pipeline for training a photocatalyst classifier using domain-specific BERT-like models (RoBERTa). This project implements a three-stage approach: **BPE Tokenizer Training**, **Language Model Pre-training**, and **Classification Fine-tuning**.

## 📋 Overview

This repository contains code to build a photocatalyst predictor that learns from chemical reactions and functional groups. The model predicts suitable photocatalysts for given chemical reactions using a transformer-based architecture trained on domain-specific data.

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: BPE Tokenizer Training                                 │
│ Input: Reactions & Functional Groups                            │
│ Output: Vocabulary & Merge Rules (BPETokenizer/)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: RoBERTa Pre-training (Masked Language Modeling)        │
│ Input: Tokenized Reactions & Groups + Trained Tokenizer        │
│ Output: Pre-trained Language Model                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Fine-tuning for Photocatalyst Classification           │
│ Input: Pre-trained Model + Labeled Catalyst Data                │
│ Output: Classification Models (5-Fold Cross-Validation)         │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Requirements

### Python Version
- Python 3.11 or higher

### Dependencies
```
numpy >= 2.3.4
scikit-learn >= 1.7.2
torch >= 2.9.0
torchmetrics >= 1.8.2
tqdm >= 4.67.1
transformers >= 4.57.1
tokenizers >= 0.13.0
```

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Photocatalyst-Recommender-based-on-a-Large-Language-Model
```

2. **Install dependencies using uv:**
```bash
uv sync
```

## 📊 Data Preparation

### Required Data Files

Before running the training pipeline, prepare the following data files:

#### 1. **pretraining_data.pickle** (For tokenizer training and pre-training)
A Python pickle file containing a dictionary with:
```python
{
    'reactions': [list of chemical reaction SMILES strings],
    'groups': [list of transformation SMILES strings]
}
```
- **Size**: ~389 MB (depending on dataset size)
- **Format**: Python pickle serialized dictionary
- **Usage**: Used in stages 1 and 2
- **Parallel structure**: `reactions[i]` and `groups[i]` are paired for each sample

**Example content:**
```python
{
    'reactions': ['B(C1CCCC1)C1CCCC1.C#CCBr>>BrC/C=C/B(C1CCCC1)C1CCCC1', ...],
    'groups': ['CC(=O)[O-].NO.O=C1CC2CC(=O)CC2C1>>Oc1cccc2cccnc12',
               'N.cc(=O)oc(n)=O>>O.cc(=O)nc(n)=O', ...]
}
```

#### 2. **Fine-tuning Data (Photocatalyst Classification)**

Due to Reaxys policy restrictions on direct data distribution, the fine-tuning dataset is not included in this repository. However, the data structure and photocatalyst information are documented here.

**Data Source**: `@data/finetunning_rxid.xlsx`
- Contains Reaxys reaction IDs with assigned photocatalyst IDs
- Used to construct the classification dataset for fine-tuning

**Expected Data Structure**:
The fine-tuning data should be a Python pickle file containing a dictionary with:
```python
{
    'reactions': [list of chemical reaction SMILES strings],
    'groups': [list of transformation SMILES strings],
    'y': [one-hot encoded catalyst labels],
    'catal_num_to_smiles_map': {catalyst_id: SMILES_string},
    'catal_smiles_to_num_map': {SMILES_string: catalyst_id}
}
```
- **Parallel structure**: `reactions[i]`, `groups[i]`, and `y[i]` correspond to the same sample
- **y format**: 2D numpy array of shape (num_samples, num_catalysts) with one-hot encoding
- **Usage**: Used in stage 3 (Fine-tuning for classification)

## 🚀 Training Pipeline

### Stage 1: BPE Tokenizer Training

**Purpose**: Train a Byte-Pair Encoding (BPE) tokenizer on your domain-specific chemical data.

**Input**: `pretraining_data.pickle`
**Output**:
- `BPETokenizer/vocab.json` - Vocabulary file with ~30K tokens
- `BPETokenizer/merges.txt` - BPE merge operations

**Command**:
```bash
uv run bert_tokenizer.py
```

**What it does**:
1. Loads reaction and functional group data from `pretraining_data.pickle`
2. Creates a temporary dataset file with whitespace-separated tokens
3. Trains a BPE tokenizer with special tokens:
   - `<UNK>` - Unknown tokens
   - `<SEP>` - Sequence separator
   - `<MASK>` - Masked token (for pre-training)
   - `<CLS>` - Classification token
4. Saves tokenizer configuration and vocabulary

**Training Configuration**:
- Vocabulary size: ~30,000 tokens
- Pre-tokenization: Whitespace-based
- Training algorithm: Byte-Pair Encoding

**Estimated Time**: 5-15 minutes depending on dataset size

### Stage 2: RoBERTa Pre-training

**Purpose**: Pre-train a RoBERTa language model using Masked Language Modeling (MLM) on chemical data.

**Input**:
- `pretraining_data.pickle`
- `BPETokenizer/` (from Stage 1)

**Output**: Trained RoBERTa model (e.g., `reactioberto_reaction_count_5000_world_len_128.pth`)

**Command**:
```bash
uv run bert_pretrain.py
```

**What it does**:
1. Loads reaction and functional group data
2. Tokenizes sequences using the trained BPE tokenizer
3. Creates a masked language modeling dataset:
   - 15% of tokens are randomly masked
   - Model learns to predict masked tokens from context
4. Trains RoBERTa model for 3 epochs
5. Saves the trained model

**Model Architecture**:
| Parameter | Value |
|-----------|-------|
| Hidden Size | 768 |
| Number of Attention Heads | 12 |
| Number of Hidden Layers | 6 |
| Max Sequence Length | 1,024 |
| Intermediate Size | 3,072 |
| Hidden Dropout Probability | 0.1 |
| Attention Dropout Probability | 0.1 |

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Batch Size | 18 |
| Number of Epochs | 3 |
| Masking Strategy | Random 15% |
| Loss Function | Cross-entropy (MLM) |

**Estimated Time**: 2-6 hours on GPU (depends on dataset size and hardware)

### Stage 3: Fine-tuning for Classification

**Purpose**: Fine-tune the pre-trained model for photocatalyst classification using 5-fold cross-validation.

**Input**:
- Pre-trained model (from Stage 2)
- Fine-tuning classification data with labeled catalysts
- `BPETokenizer/` (from Stage 1)

**Output**:
- 5 fold-specific models in `KFold/` directory
- 1 final model trained on entire dataset
- Evaluation metrics (accuracy, F1-score)

**Command**:
```bash
uv run bert_classify.py
```

**What it does**:
1. Loads the pre-trained RoBERTa model
2. Loads classification data with labeled catalysts
3. Implements 5-fold stratified cross-validation:
   - Splits data into 5 equal folds while maintaining class distribution
   - For each fold:
     - Uses 4 folds for training
     - Uses 1 fold for validation
     - Saves the best model for that fold
4. Computes metrics across all folds
5. Trains a final model on the entire dataset

**Fine-tuning Configuration**:
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size (Train) | 16 |
| Batch Size (Validation) | 6 |
| Number of Epochs | 6 |
| Loss Function | Cross-entropy |
| Evaluation Metric | Weighted F1, Macro F1, Accuracy |

**Cross-Validation**:
- **Strategy**: Stratified K-Fold (K=5)
- **Purpose**: Ensures robust evaluation and prevents overfitting
- **Output**: Per-fold metrics and averaged metrics

**Evaluation Metrics**:
- **Accuracy**: Percentage of correct predictions
- **F1 (Weighted)**: Weighted average F1 score across all classes
- **F1 (Macro)**: Unweighted average F1 score across all classes

**Model Outputs**:
```
KFold/
├── reactioberto_classify_photocatals_fold_0_val_acc_0.92_val_f1_0.89.pth
├── reactioberto_classify_photocatals_fold_1_val_acc_0.91_val_f1_0.88.pth
├── reactioberto_classify_photocatals_fold_2_val_acc_0.93_val_f1_0.91.pth
├── reactioberto_classify_photocatals_fold_3_val_acc_0.92_val_f1_0.90.pth
├── reactioberto_classify_photocatals_fold_4_val_acc_0.90_val_f1_0.87.pth
└── ...
```

**Estimated Time**: 4-12 hours on GPU (depends on dataset size, hardware, and number of classes)

## 📁 Project Structure

```
.
├── bert_tokenizer.py                 # Stage 1: Train BPE tokenizer
├── bert_pretrain.py                  # Stage 2: Pre-train RoBERTa
├── bert_classify.py                  # Stage 3: Fine-tune for classification
├── pretraining_data.pickle           # Training data (reactions + groups)
├── data/
│   └── finetunning_rxid.xlsx         # Reaxys reaction IDs with catalyst assignments
├── BPETokenizer/                     # Trained tokenizer (output of Stage 1)
│   ├── vocab.json                    # BPE vocabulary
│   └── merges.txt                    # BPE merge rules
├── KFold/                            # Fold-specific models (output of Stage 3)
├── pyproject.toml                    # Project configuration
├── .python-version                   # Python version specification
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

## 🎯 Complete Training Workflow

### Quick Start (All Stages)

```bash
# Stage 1: Train Tokenizer (5-15 minutes)
uv run bert_tokenizer.py

# Stage 2: Pre-train Model (2-6 hours)
uv run bert_pretrain.py

# Stage 3: Fine-tune for Classification (4-12 hours)
uv run bert_classify.py
```
## 💡 Key Concepts

### Byte-Pair Encoding (BPE)
- Tokenization algorithm that iteratively merges the most frequent character pairs
- Reduces vocabulary size while maintaining information
- Effective for domain-specific vocabularies (chemical SMILES)

### Masked Language Modeling (MLM)
- Pre-training objective where random tokens are masked
- Model learns to predict masked tokens from surrounding context
- Enables transfer learning and domain adaptation

### RoBERTa
- **Ro**bustly **o**ptimized **BERT** **a**pproach
- Improvements over original BERT:
  - Better pre-training strategy
  - Longer training time
  - Improved masking patterns
  - Larger batches

### Stratified K-Fold Cross-Validation
- Divides data into K equal-sized folds
- Maintains class distribution in each fold
- Prevents data leakage and biased evaluation
- Provides multiple trained models for ensemble methods

## 🔍 Monitoring Training

### Progress Indicators

During training, you'll see:
- Progress bars with iterations, loss, and ETA
- Validation metrics at regular intervals
- Best model checkpoints automatically saved

### Log Output Example

```
[Stage 2: Pre-training]
Epoch 1/3: 45%|████▌| 450/1000 [12:30<15:15, 0.67 it/s, loss=2.34]
[Stage 3: Fine-tuning Fold 1]
Epoch 2/6: 100%|██████| 100/100 [03:45<00:00, 0.44 it/s, val_acc=0.92, val_f1=0.89]
```

## 🛠️ Advanced Usage

### Custom Configuration

To modify training parameters, edit the corresponding script:

**bert_pretrain.py**:
```python
# Change learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Change batch size
batch_size = 32

# Change number of epochs
epochs = 5
```

**bert_classify.py**:
```python
# Change number of folds
n_splits = 10

# Change learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Change batch size
train_batch_size = 32
```
## 📝 Data Format Details

### Reaction SMILES Format

Reactions are represented as SMILES strings with reactants and products:
```
reactants>>products
C=C>>C(C)C          # Propene to propane
c1ccccc1>>c1ccccc1C # Benzene to toluene
```

### Transformation SMILES Strings (Groups)

In the `groups` field, transformations are represented as SMILES reaction strings (similar to reactions but representing different chemical transformations):
```
CC(=O)[O-].NO.O=C1CC2CC(=O)CC2C1>>Oc1cccc2cccnc12
N.cc(=O)oc(n)=O>>O.cc(=O)nc(n)=O
C1CC[Si]C1>>C1=C[Si]C=C1
CN.CSC(=C)SC.N>>C.C.CNC(=C)N.S.S
cB(O)O.ccc.n>>c.cc(c)-n
```

These represent chemical transformations or reaction fragments that pair with the main reactions in the parallel `reactions` list.

### Photocatalyst ID to SMILES Mapping

The fine-tuning dataset includes 31 photocatalysts identified by numeric IDs. The following table maps each ID to its corresponding SMILES string:

| ID | SMILES |
|---|---|
| 0 | CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.c1ccc(-c2ccccc2[Ir]c2ccccc2-c2ccccn2)nc1 |
| 1 | [Ru].c1cnc2c(c1)ccc1cccnc12.c1cnc2c(c1)ccc1cccnc12.c1cnc2c(c1)ccc1cccnc12 |
| 2 | c1ccc(-c2ccccc2[Ir](c2ccccc2-c2ccccn2)c2ccccc2-c2ccccn2)nc1 |
| 3 | [Ru].c1ccc(-c2ccccn2)nc1.c1ccc(-c2ccccn2)nc1.c1ccc(-c2ccccn2)nc1 |
| 4 | COc1ccc(-c2ccc3ccc4ccc(-c5ccc(OC)cc5)nc4c3n2)cc1.COc1ccc(-c2ccc3ccc4ccc(-c5ccc(OC)cc5)nc4c3n2)cc1.[Cu+] |
| 5 | Fc1ccc(-c2ccccn2)c(F)c1.Fc1ccc(-c2ccccn2)c(F)c1.Fc1ccc(-c2ccccn2)c(F)c1.[Ir] |
| 6 | CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.Fc1ccc(-c2ccc(C(F)(F)F)cn2)c(F)c1.Fc1ccc(-c2ccc(C(F)(F)F)cn2)c(F)c1.[Ir+] |
| 7 | CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21.Cc1ccc2ccc3ccc(C)nc3c2n1.[Cu+] |
| 8 | [Ru].c1cnc(-c2cnccn2)cn1.c1cnc(-c2cnccn2)cn1.c1cnc(-c2cnccn2)cn1 |
| 9 | CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.[Ru] |
| 10 | N#Cc1c(N(c2ccccc2)c2ccccc2)c(C#N)c(N(c2ccccc2)c2ccccc2)c(N(c2ccccc2)c2ccccc2)c1N(c1ccccc1)c1ccccc1 |
| 11 | O=C1c2ccccc2C(=O)c2ccccc21 |
| 12 | CCN(CC)c1ccc2c(-c3ccccc3C(=O)O)c3ccc(=[N+](CC)CC)cc-3oc2c1 |
| 13 | c1ccc(-c2cc(-c3ccccc3)[o+]c(-c3ccccc3)c2)cc1 |
| 14 | O=C1C(Cl)=C(Cl)C(=O)C(Cl)=C1Cl |
| 15 | O=C1c2ccccc2-c2ccccc21 |
| 16 | N#Cc1c2ccccc2c(C#N)c2ccccc12 |
| 17 | N#Cc1c(-n2c3ccccc3c3ccccc32)c(C#N)c(-n2c3ccccc3c3ccccc32)c(-n2c3ccccc3c3ccccc32)c1-n1c2ccccc2c2ccccc21 |
| 18 | c1ccc(N2c3ccccc3Sc3ccccc32)cc1 |
| 19 | O=C1OC2(c3ccc(O)cc3Oc3cc(O)ccc32)c2ccccc21 |
| 20 | N#Cc1ccc(C#N)c2ccccc12 |
| 21 | O=C([O-])c1c(Cl)c(Cl)c(Cl)c(Cl)c1-c1c2cc(I)c(=O)c(I)c-2oc2c(I)c([O-])c(I)cc12 |
| 22 | CN(C)c1ccc2nc3ccc(=[N+](C)C)cc-3sc2c1 |
| 23 | CN(C)c1ccc(C(=O)c2ccc(N(C)C)cc2)cc1 |
| 24 | O=c1c2ccccc2sc2ccccc12 |
| 25 | Cc1cc(C)c(-c2c3ccccc3[n+](C)c3ccccc23)c(C)c1 |
| 26 | O=c1c2ccccc2oc2ccccc12 |
| 27 | O=C(c1ccccc1)c1ccccc1 |
| 28 | N#Cc1ccc(C#N)cc1 |
| 29 | CCNc1cc2oc3cc(=[NH+]CC)c(C)cc-3c(-c3ccccc3C(=O)OCC)c2cc1C |
| 30 | O=C(O)c1ccccc1-c1c2cc(Br)c(=O)c(Br)c-2oc2c(Br)c(O)c(Br)cc12 |

**Total number of photocatalysts**: 31

These catalysts include a variety of organic dyes, metal complexes, and heterogeneous photocatalysts commonly used in organic synthesis.

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Byte Pair Encoding (BPE)](https://en.wikipedia.org/wiki/Byte_pair_encoding)
