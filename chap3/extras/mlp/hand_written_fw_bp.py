import torch
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.cos(x)

x_train = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

class SimpleMLP:
    def __init__(self):
        self.w1 = torch.randn(1, 100, dtype=torch.float32) * 0.1
        self.b1 = torch.zeros(100, dtype=torch.float32)
        self.w2 = torch.randn(100, 100, dtype=torch.float32) * 0.1
        self.b2 = torch.zeros(100, dtype=torch.float32)
        self.w3 = torch.randn(100, 1, dtype=torch.float32) * 0.1
        self.b3 = torch.zeros(1, dtype=torch.float32)

    def forward(self, x):
        self.z1 = torch.matmul(x, self.w1) + self.b1
        self.a1 = torch.tanh(self.z1)
        self.z2 = torch.matmul(self.a1, self.w2) + self.b2
        self.a2 = torch.tanh(self.z2)
        self.z3 = torch.matmul(self.a2, self.w3) + self.b3
        return self.z3

model = SimpleMLP()

lr = 0.01
epochs = 10000

''' 

z1 = x @ w1 + b1
a1 = tanh(z1)
z2 = a1 @ w2 + b2
a2 = tanh(z2)
z3 = a2 @ w3 + b3

loss = (z3 - target)**2/target.size(0)

dl/dz3 = 2*(z3 - target)/target.size(0)

dl/dw3 = a2.t @ dl/dz3 
dl/a2 = dl/dz3*dz3/da = dl/dz3 @ (w3.t)

dl/dz2 = dl/da2 * (1 - a**2)

dl/a1 = dl/dz2 * (w2.t)

...

'''

for epoch in range(epochs):
    y_pred = model.forward(x_train)
    loss = ((y_pred - y_train) ** 2).mean()

    grad_z3 = 2 * (y_pred - y_train) / y_train.size(0)
    grad_w3 = torch.matmul(model.a2.t(), grad_z3)
    grad_b3 = grad_z3.sum(0)

    grad_a2 = torch.matmul(grad_z3, model.w3.t())
    grad_z2 = grad_a2 * (1 - model.a2 ** 2)
    grad_w2 = torch.matmul(model.a1.t(), grad_z2)
    grad_b2 = grad_z2.sum(0)

    grad_a1 = torch.matmul(grad_z2, model.w2.t())
    grad_z1 = grad_a1 * (1 - model.a1 ** 2)
    grad_w1 = torch.matmul(x_train.t(), grad_z1)
    grad_b1 = grad_z1.sum(0)

    with torch.no_grad():
        model.w1 -= lr * grad_w1
        model.b1 -= lr * grad_b1
        model.w2 -= lr * grad_w2
        model.b2 -= lr * grad_b2
        model.w3 -= lr * grad_w3
        model.b3 -= lr * grad_b3

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    y_pred = model.forward(x_train)

plt.plot(x, y, label='True cos(x)')
plt.plot(x, y_pred.numpy(), label='MLP prediction')
plt.legend()
plt.show()
