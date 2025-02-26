import torch
from model import TinyTransformer
from data import prepare_data

# Load data and model
print("Loading model...")
input_data, vocab_size, word_to_idx, idx_to_word = prepare_data()
model = TinyTransformer(vocab_size=vocab_size)
model.load_state_dict(torch.load("my_llm_model.pth"))
print("Model loaded successfully!")

def generate_text(seed_text, length=10, temperature=0.8):
    model.eval()
    words = seed_text.lower().split()
    
    for _ in range(length):
        # Get the last 8 words or fewer if we don't have 8 yet
        context = words[-8:] if len(words) >= 8 else words
        
        # Convert words to indices
        indices = [word_to_idx.get(word, 0) for word in context]
        
        # Pad if needed
        if len(indices) < 8:
            indices = [0] * (8 - len(indices)) + indices
            
        x = torch.tensor(indices).unsqueeze(0)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            output = model(x)
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_word_idx = torch.multinomial(probs, 1).item()
        
        next_word = idx_to_word[next_word_idx]
        words.append(next_word)
    
    return " ".join(words)

# Interactive loop
print("\nWelcome to your Interactive LLM Demo!")
print("Type 'exit' to quit.\n")

while True:
    seed = input("\nEnter a starting phrase: ")
    if seed.lower() == 'exit':
        break
    
    try:
        length = int(input("How many words to generate: "))
        temp = float(input("Temperature (0.5-1.5): "))
        print("\nGenerating...")
        result = generate_text(seed, length, temp)
        print(f"\nGenerated text: {result}")
    except ValueError:
        print("Please enter valid numbers for length and temperature")
    except Exception as e:
        print(f"An error occurred: {e}")

print("Thank you for using the demo!")