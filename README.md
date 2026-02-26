# SentimentScope: Transformer-Based Sentiment Analysis

A custom transformer model built from scratch for binary sentiment classification on the IMDB movie review dataset.

## Project Overview

This project implements a GPT-style transformer architecture for sentiment analysis, achieving **76.28% accuracy** on the IMDB test set. The model is trained entirely from scratch without using pretrained weights.

## Performance

- **Test Accuracy**: 76.28%
- **Validation Accuracy**: 77.12%
- **Training Time**: 3 epochs
- **Dataset**: 50,000 IMDB movie reviews

## Architecture

- **Model Type**: Custom GPT-style Transformer
- **Layers**: 4 transformer blocks
- **Attention Heads**: 4 heads per block
- **Embedding Dimension**: 128
- **Head Dimension**: 32
- **Context Length**: 128 tokens
- **Tokenizer**: BERT base uncased

### Key Components

1. **Multi-Head Self-Attention**: Captures contextual relationships between tokens
2. **Feed-Forward Networks**: Processes attention outputs with GELU activation
3. **Layer Normalization**: Stabilizes training
4. **Mean Pooling**: Aggregates token embeddings for classification
5. **Binary Classifier**: Final linear layer for sentiment prediction

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training

```python
# The notebook handles everything:
# 1. Downloads IMDB dataset automatically
# 2. Preprocesses and tokenizes data
# 3. Trains the model
# 4. Evaluates on test set
```

### Usage

```python
from transformers import AutoTokenizer
import torch

# Load model
model = DemoGPT(config)
model.load_state_dict(torch.load('sentiment_model.pth'))
model.eval()

# Tokenize input
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt", max_length=128, 
                   truncation=True, padding="max_length")

# Predict
with torch.no_grad():
    logits = model(inputs['input_ids'])
    prediction = torch.argmax(logits, dim=1)
    sentiment = "Positive" if prediction == 1 else "Negative"
```

## Project Structure

```
├── SentimentScope.ipynb          # Main notebook (clean, portfolio version)
├── sentiment_model.pth           # Trained model weights
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Technical Details

### Data Processing
- **Tokenization**: BERT WordPiece tokenizer
- **Max Length**: 128 tokens (truncation/padding applied)
- **Train/Val Split**: 90/10
- **Batch Size**: 32

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 3e-4
- **Loss Function**: Cross-Entropy
- **Dropout**: 0.1
- **Epochs**: 3

### Model Parameters
- **Total Parameters**: ~2.5M (estimated)
- **Vocabulary Size**: 30,522 (BERT vocab)

## Results Analysis

The model demonstrates strong generalization with minimal overfitting:
- Validation and test accuracy within 1% (77.12% vs 76.28%)
- Consistent loss reduction across epochs
- Efficient training with only 3 epochs needed

## Future Improvements

- Increase model capacity (more layers, larger embeddings)
- Implement learning rate scheduling
- Add gradient clipping for stability
- Experiment with different pooling strategies (CLS token, max pooling)
- Fine-tune pretrained BERT/RoBERTa for comparison
- Implement attention visualization
- Add model interpretability (LIME/SHAP)
- Extend to multi-class sentiment (1-5 stars)

## Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Tokenization
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

## Acknowledgments

This project was completed as part of the **AWS AI Scientist Nanodegree Program** by AWS/Udacity.

## Author

**Aakash Upadhyay**
- GitHub: [@Aakash-U](https://github.com/Aakash-U)
- LinkedIn: [Aakash Upadhyay](https://www.linkedin.com/in/aakash-upadhyay-524347137/)

---

**Note**: This is an educational project demonstrating transformer architecture for sentiment analysis. The implementation showcases practical application of deep learning concepts in NLP.
