import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import os

torch.manual_seed(0)
np.random.seed(0)

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def generate_lorenz_data(t_max=4.0, dt=0.01):
    steps = int(t_max / dt)
    states = np.zeros((steps, 3))
    states[0] = [1.0, 1.0, 1.0]
    for i in range(1, steps):
        states[i] = states[i - 1] + dt * lorenz(i * dt, states[i - 1])
    t = np.linspace(0.0, t_max, steps)
    return t, states

t, states = generate_lorenz_data()
os.makedirs("lorenz_output", exist_ok=True)
np.save("lorenz_output/lorenz_t.npy", t)
np.save("lorenz_output/lorenz_states.npy", states)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, t):
        return self.net(t)

def pinn_loss(model, t, sigma, rho, beta):
    t = t.view(-1, 1)
    t.requires_grad = True
    pred = model(t)
    x, y, z = pred[:, 0], pred[:, 1], pred[:, 2]
    dx = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dy = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    dz = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True)[0]
    eq1 = dx - sigma * (y - x)
    eq2 = dy - (x * (rho - z) - y)
    eq3 = dz - (x * y - beta * z)
    return (eq1**2 + eq2**2 + eq3**2).mean()

def train_pinn(t, sigma, rho, beta, epochs=3000):
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    t_tensor = torch.tensor(t, dtype=torch.float32)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pinn_loss(model, t_tensor, sigma, rho, beta)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"[PINN] Epoch {epoch}, Loss: {loss.item():.6f}")
    torch.save(model.state_dict(), "lorenz_output/pinn_model.pth")
    return model

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, t, x):
        return self.net(x)

def train_neural_ode(t, states, epochs=5000):
    func = ODEFunc()
    optimizer = optim.Adam(func.parameters(), lr=0.001)
    x0 = torch.tensor(states[0], dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)
    y_true = torch.tensor(states, dtype=torch.float32)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_y = odeint(func, x0, t_tensor)
        loss = torch.mean((pred_y - y_true) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"[ODE] Epoch {epoch}, Loss: {loss.item():.6f}")
    torch.save(func.state_dict(), "lorenz_output/neural_ode_model.pth")
    return func

def plot_results(t, states, pinn_model, neural_ode_model):
    t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        pinn_pred = pinn_model(t_tensor).numpy()
        x0 = torch.tensor(states[0], dtype=torch.float32)
        pred_y = odeint(neural_ode_model, x0, torch.tensor(t, dtype=torch.float32))
        ode_pred = pred_y.numpy()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(states[:, 0], states[:, 1], states[:, 2], label="True", linewidth=1)
    ax1.plot(pinn_pred[:, 0], pinn_pred[:, 1], pinn_pred[:, 2], label="PINN", linewidth=1)
    ax1.set_title("PINN vs True")
    ax1.legend()
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(states[:, 0], states[:, 1], states[:, 2], label="True", linewidth=1)
    ax2.plot(ode_pred[:, 0], ode_pred[:, 1], ode_pred[:, 2], label="Neural ODE", linewidth=1)
    ax2.set_title("Neural ODE vs True")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("lorenz_output/results.png")
    plt.show()

if __name__ == "__main__":
    pinn_model = train_pinn(t, sigma, rho, beta)
    neural_ode_model = train_neural_ode(t, states)
    plot_results(t, states, pinn_model, neural_ode_model)