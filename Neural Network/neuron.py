import numpy as np
np.random.seed(0)

X = np.array([[0,0],[0,1],[1,0],[1,2]])
y = np.array([[0],[1],[1],[0]])

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(a): 
    return a*(1-a)

W1 = np.random.randn(2,4); b1 = np.zeros((1, 4))
W2 = np.random.randn(4,1); b2 = np.zeros((1,1))
lr = 0.5

for epoch in range(10000):
    hidden = sigmoid(X @ W1 + b1)
    out = sigmoid(hidden @ W2 + b2)

    loss = np.mean((out - y) ** 2)

    d_out = (out - y) * sigmoid_deriv(out)
    d_hidden = (d_out @ W2.T) * sigmoid_deriv(hidden)

    if epoch % 2000 == 0:
        print(f'epoch {epoch}, loss {loss:.4f}')

print("Predictions:\n", out.round(3))