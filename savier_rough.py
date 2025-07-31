import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

# Modular deep network
class DeepNN(nn.Module):
    def __init__(self, activation_fn, init_method):
        super(DeepNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            activation_fn(),
            nn.Linear(64, 64),
            activation_fn(),
            nn.Linear(64, 2)
        )
        self.init_method = init_method
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                if self.init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_method == 'random':
                    nn.init.uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)

# Training loop
def train_model(model, epochs=100, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train_tensor)
        loss = criterion(out, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history

# Variants to compare
variants = [
    ("ReLU + Kaiming", nn.ReLU, "kaiming"),
    ("Tanh + Xavier", nn.Tanh, "xavier"),
    ("ReLU + Xavier", nn.ReLU, "xavier"),
    ("ReLU + Random", nn.ReLU, "random")
]

# Train and plot
plt.figure(figsize=(12, 6))
for name, act_fn, init in variants:
    model = DeepNN(act_fn, init)
    losses = train_model(model)
    plt.plot(losses, label=name)

plt.title("Loss Comparison: Activation + Initialization")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
