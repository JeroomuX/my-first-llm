import torch
from model import TinyTransformer
from data import prepare_data

print("Loading model...")
input_data, vocab_size, word_to_idx, idx_to_word = prepare_data()
model = TinyTransformer(vocab_size=vocab_size)
model.load_state_dict(torch.load("my_llm_model.pth"))

print("Model loaded successfully!")
print("Generating sample text...")
# Simple test generation
model.eval()
seed_text = "Python is"
length = 10
words = seed_text.lower().split()

for _ in range(length):
    x = torch.tensor([word_to_idx.get(word, 0) for word in words[-8:]])
    if len(x) < 8:  # Pad if needed
        x = torch.cat([torch.zeros(8-len(x), dtype=torch.long), x])
    x = x.unsqueeze(0)
    
    with torch.no_grad():
        output = model(x)
        logits = output[0, -1] / 0.8  # temperature
        probs = torch.softmax(logits, dim=0)
        next_word_idx = torch.multinomial(probs, 1).item()
    
    next_word = idx_to_word[next_word_idx]
    words.append(next_word)

result = " ".join(words)
print(f"Result: {result}")
print("Done!")