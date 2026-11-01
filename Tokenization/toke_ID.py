import tiktoken

enc = tiktoken.get_encoding("o200k_base")

text = "My Name is Ravi!"

token_id = enc.encode(text)
print(f'token Id"s"{token_id}')

pieces = [enc.decode([t]) for t in token_id]
print(f'pieces: {pieces}')

print(f'Decoded: {enc.decode(token_id)}')