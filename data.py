import torch

# Simple dataset for testing
training_text = """
Python is a great programming language.
I love coding and building projects.
Learning machine learning is exciting.
Neural networks can recognize patterns.
Data science helps us understand information.
Programming requires practice and patience.
Building models is a creative process.
Code can solve many different problems.
"""

def prepare_data():
    # Simple tokenization by splitting on spaces and lowercase
    tokens = training_text.lower().split()
    
    # Create vocabulary
    vocab = sorted(set(tokens))
    vocab_size = len(vocab)
    
    # Create mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Convert text to numbers
    numeric_text = [word_to_idx[word] for word in tokens]
    
    return torch.tensor(numeric_text), vocab_size, word_to_idx, idx_to_word