# N-gram Language Model

A Python-based N-gram language model implementation that learns from Jane Austen's "Emma" text and generates new sentences using probabilistic prediction.

## Table of Contents

- [What is an N-gram Language Model?](#what-is-an-n-gram-language-model)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Model Architecture](#model-architecture)
- [Examples](#examples)
- [Dependencies](#dependencies)

---

## What is an N-gram Language Model?

An **N-gram** is a sequence of N consecutive words from text. An N-gram language model learns patterns in text by counting how often certain word sequences appear together, then uses these patterns to predict the next word given a context.

### How it Works

1. **Training Phase**:
   - Break text into N-grams (e.g., 3-grams: ["the", "quick", "brown"])
   - Count occurrences of each N-gram
   - Store context (first N-1 words) and their following words
   - Calculate probabilities: P(word | context)

2. **Prediction Phase**:
   - Given a context (e.g., "what is"), look up all possible next words
   - Sort by probability
   - Select the most likely word
   - Update context and repeat

### Example with 3-grams

Text: "the cat sat on the mat"

3-grams:
- ("the", "cat", "sat")
- ("cat", "sat", "on")
- ("sat", "on", "the")
- ("on", "the", "mat")

To predict: context = ("the", "mat"), the model looks at all 3-grams starting with ("the", "mat") and returns the most probable next word.

---

## Project Overview

This project implements a **3-gram language model** trained on Jane Austen's "Emma". It demonstrates:

- **Text Tokenization**: Cleaning and splitting raw text into words
- **Probability Calculation**: Computing P(word | context) with optional Laplace smoothing
- **Text Generation**: Using the model to generate coherent sequences
- **Interactive Output**: Real-time word generation with visual feedback

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Clone the Repository

```bash
git clone https://github.com/yourusername/n-gram-language-model.git
cd n-gram-language-model
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install regex nltk
```

### Download Required Data

The first time you run the script, it will automatically download Jane Austen's "Emma" from NLTK:

```bash
python main.py
```

---

## Usage

### Running the Text Generator

```bash
python main.py
```

This will:
1. Load the trained N-gram model
2. Start with context ("what", "is")
3. Generate 20 words with 0.25-second delay between each word
4. Clear the console and display the growing sentence in real-time
5. Print the final paragraph

### Example Output

```
what is
what is the
what is the most
what is the most important
what is the most important thing
...
```

### Using the Model Programmatically

```python
from model import build_model

# Build and train the model
model, tokens = build_model(n=3, smoothing=False)

# Define starting context
start_context = ("what", "is")

# Get predictions for a context
predictions = model.predict(start_context)
# Returns: [{'most': 0.45}, {'very': 0.30}, {'quite': 0.25}]

# Get probability of a word given context
prob = model.get_probability(start_context, "most")
# Returns: 0.45
```

---

## Files

### `model.py`
Contains the core N-gram implementation:

- **`tokenize(text)`**: Cleans text by removing punctuation, converting to lowercase, and splitting into words
  - Regex pattern: `r'[^\w\s]'` removes anything that isn't a word character or whitespace
  
- **`NGramModel` class**: Main model class with methods:
  - `train(tokens)`: Learn patterns from tokenized text
  - `predict(context)`: Return probable next words sorted by likelihood
  - `get_probability(context, word)`: Calculate P(word | context)

- **`build_model(n, smoothing)`**: Helper function to download, tokenize, and train the model

### `main.py`
Demonstrates the model in action:

- **`generate_paragraph()`**: Generates a sequence of words with:
  - Real-time console clearing (`os.system("cls")`)
  - Time delay between words (`time.sleep(0.25)`)
  - Visual feedback showing the growing sentence

### `model.ipynb`
Jupyter notebook with step-by-step model exploration and testing

---

## Model Architecture

### NGramModel Class

```
NGramModel(n=3, smoothing=False)
│
├── ngrams: Counter
│   └── Stores count of each n-gram
│
├── contexts: defaultdict(Counter)
│   └── Maps context → {word: count}
│
├── context_count: Counter
│   └── Stores total count for each context
│
├── vocab: set
│   └── Unique words in training data
│
└── smoothing: bool
    └── Whether to apply Laplace smoothing
```

### Probability Calculation

**Without smoothing**:
```
P(word | context) = count(context, word) / count(context)
```

**With Laplace smoothing** (handles unseen words):
```
P(word | context) = (count(context, word) + 1) / (count(context) + |vocab|)
```

---

## Examples

### Example 1: Generate from "what is"

```python
from model import build_model

model, tokens = build_model(n=3, smoothing=False)
context = ("what", "is")
predictions = model.predict(context)
predictions.sort(key=lambda x: list(x.values())[0], reverse=True)
print(predictions[:3])
# Output: [{'most': 0.45}, {'the': 0.25}, {'very': 0.15}]
```

### Example 2: Use Smoothing for Rare Words

```python
model_smooth, tokens = build_model(n=3, smoothing=True)
prob = model_smooth.get_probability(("rare", "context"), "word")
# Even if combination hasn't been seen, will return a small probability
```

### Example 3: Interactive Text Generation

```python
from model import build_model

model, tokens = build_model(n=3, smoothing=False)

# Custom text generation
context = ("i", "am")
for i in range(10):
    predictions = model.predict(context)
    if predictions:
        predictions.sort(key=lambda x: list(x.values())[0], reverse=True)
        next_word = list(predictions[0].keys())[0]
        print(next_word, end=" ")
        context = (context[1], next_word)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `regex` | Latest | Advanced text pattern matching for tokenization |
| `nltk` | Latest | Natural Language Toolkit for downloading corpus |
| `collections` | Built-in | Counter and defaultdict for data structures |

### Install All

```bash
pip install regex nltk
```

---

## Key Concepts Explained

### Tokenization

The `tokenize()` function prepares raw text:

```python
def tokenize(text):
    text = text.strip()  # Remove leading/trailing whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Normalize case
    return text.split()  # Split on whitespace
```

**Input**: `"Hello, World! This is a test."`
**Output**: `["hello", "world", "this", "is", "a", "test"]`

### Lambda Functions in Sorting

```python
predictions.sort(key=lambda x: list(x.values())[0], reverse=True)
```

- `lambda x: ...` is an anonymous function
- `list(x.values())[0]` extracts the probability from a dict like `{"word": 0.45}`
- `reverse=True` sorts highest probability first

### Context Windows

For a 3-gram model:
- **Context**: First 2 words
- **Next Word**: 3rd word

This sliding window moves through the text to learn transitions:
```
[the] [cat] [sat]  →  context=("the", "cat"), next="sat"
      [cat] [sat] [on]  →  context=("cat", "sat"), next="on"
```

---

## Performance Notes

- **Model Size**: ~35,000 unique 3-grams from "Emma"
- **Vocabulary**: ~7,000 unique words
- **Training Time**: ~2-3 seconds on first run (includes NLTK download)
- **Generation Speed**: Instantaneous predictions

---

## Future Enhancements

- Larger N-grams (4-grams, 5-grams)
- Multiple source texts
- Backoff strategies for unseen contexts
- Beam search for better coherence
- Web interface for text generation

---

## License

This project is open source. Feel free to modify and distribute.

---



