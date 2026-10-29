import torch

x = torch.tensor(5.0, requires_grad=True)

y = x**3 - 4*x**2 + x

print(f"x = {x.item()}")
print(f"y = x³ - 4x² + x = {y.item()}")

y.backward()

print(f"dy/dx = {x.grad.item()}")