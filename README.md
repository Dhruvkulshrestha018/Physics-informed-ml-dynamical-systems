# Research on PIML based technique for complex Dynamical systems  
_A study of physics-informed and data-driven models for nonlinear dynamical systems_

This repository contains the complete code, results, and experiments for a comparative study of several scientific machine-learning methods applied to nonlinear dynamical systems such as the **Lorenz system** and **1D Burgers' equation**.

The goal of the project is to understand how different modeling approaches perform in terms of:
- Prediction accuracy  
- Stability on chaotic systems  
- Data efficiency  
- Ability to capture underlying physics  

---

## 🚀 **Models Compared**
This study evaluates the following modern approaches:

### **1. Physics-Informed Neural Networks (PINN)**
- Trains neural networks using PDE/ODE residuals.
- Good for physics-constrained regression.
- Implemented using PyTorch.

### **2. Neural ODEs**
- Continuous-time neural modeling.
- Learns dynamics directly from data.
- Implemented using `torchdiffeq`.

### **3. Fourier Neural Operator (FNO)**
- Operator-learning method for PDEs.
- Fast and accurate for spatial-temporal systems.

### **4. SINDy (Sparse Identification of Nonlinear Dynamics)**
- Symbolic regression for discovering governing equations.
- Produces
comparitive_study_PINN_NeuralODE/
│── lorenz_output/ # trained models + results
│ ├── lorenz_states.npy
│ ├── lorenz_t.npy
│ ├── neural_ode_model.pth
│ ├── pinn_model.pth
│ ├── results.png
│ └── PIML_finalpaper.pdf # research paper
│
│── lorenz_pinn_ode.py # main comparative script
│── requirements.txt
│
Models and data/
│── burgers_pop.ipynb # Burgers' equation experiments
│── cleaned_dataset_for_neural_ode.csv
│
│── FNO/
│ ├── burgers_u.npy
│ ├── burgers_x.npy
│ ├── fno1d_burgers.pth
│ └── fno1d.ipynb
│
│── Neurla ODE/
│ └── NODE3.ipynb
│
│── PINN/
│ ├── PINN.ipynb
│ └── burgers0.h5
│
│── SINDY/
│ └── SINDy3.ipynb
│
results_of_all_models/ # output plots & comparisons
torch_diffeq/
│── PINNmodel.ipynb # alternative PINN + ODE testing


---

## 📊 **Experiments**
The experiments focus on:

### **Lorenz System**
- Training PINN and Neural ODE to predict chaotic trajectories.
- Visualizing long-term divergence and stability.
- Comparing learned dynamics vs true attractor.

### **1D Burgers' Equation**
- Solved using:
  - PINN  
  - FNO  
  - Neural ODE (data-driven)  
- Comparing shock capturing ability.

### **Equation Discovery (SINDy)**
- Automatically discovering underlying PDE/ODE structure.
- Comparing discovered terms to ground-truth equations.

---

## 📝 **How to Run**

Install dependencies:

```bash
pip install -r requirements.txt

