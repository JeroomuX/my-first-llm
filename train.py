import torch
import torch.nn as nn
from model import TinyTransformer
from data import prepare_data

# Get our real data
input_data, vocab_size, word_to_idx, idx_to_word = prepare_data()

# Training settings
sequence_length = 8  # Look at 8 words at a time
batch_size = 4
learning_rate = 0.001

# Initialize the model with our actual vocabulary size
model = TinyTransformer(vocab_size=vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create real training batches
def get_batch():
    # Random starting points in our text
    random_starts = torch.randint(0, len(input_data) - sequence_length - 1, (batch_size,))
    x = torch.stack([input_data[i:i+sequence_length] for i in random_starts])
    y = torch.stack([input_data[i+1:i+sequence_length+1] for i in random_starts])
    return x, y

# Training loop
def train_step():
    model.train()
    x, y = get_batch()
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Train for 100 steps
print("Starting training...")
for step in range(200):
    loss = train_step()
    if (step + 1) % 10 == 0:  # Print every 10 steps
        print(f"Step {step+1}, Loss: {loss:.4f}")

# Test the model
def generate_text(seed_text="The", length=10):
    model.eval()
    words = seed_text.lower().split()
    for _ in range(length):
        # Convert words to indices
        x = torch.tensor([word_to_idx.get(word, 0) for word in words[-sequence_length:]])
        x = x.unsqueeze(0)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            output = model(x)
            next_word_idx = output[0, -1].argmax().item()
        
        # Convert back to word and append
        next_word = idx_to_word[next_word_idx]
        words.append(next_word)
    
    return " ".join(words)

# Generate some text
print("\nGenerating text...")
generated = generate_text("Python is", length=10)
print(generated)