import torch

# Simple dataset for testing
training_text = """
Python is a versatile programming language used by beginners and experts.
Many data scientists prefer Python for its clear syntax and powerful libraries.
Machine learning models need data to learn patterns effectively.
Neural networks can solve complex problems through training.
Programming is both a science and an art that requires practice.
Good code should be readable, efficient, and well-documented.
Software development involves planning, coding, testing, and maintenance.
Computers process information based on the instructions we provide.
Learning to code opens up many career opportunities in technology.
Algorithms are step-by-step procedures for solving specific problems.
Data structures help organize information in efficient ways.
Python libraries like NumPy and Pandas make data analysis easier.
TensorFlow and PyTorch are popular frameworks for building neural networks.

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