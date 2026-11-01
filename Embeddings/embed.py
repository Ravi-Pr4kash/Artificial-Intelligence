import torch
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=10,
    embedding_dim=4
)

print(embedding.weight)

token_ids = torch.tensor([2,5,7])

vectors = embedding(token_ids)

# print("Token IDs:")
# print(token_ids)

# print("\nEmbedding vectors:")
# print(vectors)