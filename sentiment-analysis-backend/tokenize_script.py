import sys
import json
from transformers import AutoTokenizer

# Load the tokenizer from the directory passed as the first argument
tokenizer_dir = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# Read all lines from standard input
for line in sys.stdin:
    # Remove leading/trailing whitespace
    text_input = line.strip()
    if not text_input:
        continue
    
    # Tokenize the text
    token_ids = tokenizer.encode(text_input, add_special_tokens=True)
    
    # Print the token IDs as a comma-separated string
    print(",".join(map(str, token_ids)))
