import tiktoken
import numpy as np

tokenizer = tiktoken.get_encoding("cl100k_base")

text = "Hello world"
tokens = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Number of Tokens: {len(tokens)}")

print(f"\nToken Breakdown:")
for token in tokens:
    print(f" {token} -> {tokenizer.decode([token])}")