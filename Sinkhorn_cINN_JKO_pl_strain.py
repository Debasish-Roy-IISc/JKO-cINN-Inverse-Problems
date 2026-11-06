#!/usr/bin/env python
# coding: utf-8

# # batched implementation

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from geomloss import SamplesLoss
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)
# ==========================================
# 1. BATCHED MESH & MATERIAL CREATION
# ==========================================
NX, NY = 16, 16
LX, LY = 1.0, 1.0
dx, dy = LX / (NX - 1), LY / (NY - 1)
E_MIN = 100e9
E_MAX = 300e9

# Generate nodes
y_grid, x_grid = torch.meshgrid(
    torch.linspace(0, LY, NY, device=device),
    torch.linspace(0, LX, NX, device=device),
    indexing="ij"
)
nodes = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
N_NODES = nodes.shape[0]

# Generate elements (vectorized)
i_idx, j_idx = torch.meshgrid(torch.arange(NX - 1, device=device), torch.arange(NY - 1, device=device), indexing="ij")
n1 = j_idx * NX + i_idx
n2 = n1 + 1
n3 = n2 + NX
n4 = n1 + NX
elements = torch.stack([n1.flatten(), n2.flatten(), n3.flatten(), n4.flatten()], dim=1)
N_ELEMS = elements.shape[0]

# Baseline Young's Modulus in Pascals (e.g., Steel = 200 GPa)
E0 = 200e9  
NU0 = 0.30

def nodal_E_to_element_E(E_nodes):
    # E_nodes: [B, N_NODES]
    # elementwise average of the 4 corner nodal values
    return E_nodes[:, elements].mean(dim=-1) # Shapes: [B, N_ELEMS]

def elasticity_matrix_batched(E):
    nu = NU0

    coeff = E / ((1 + nu) * (1 - 2 * nu))

    B_dim, E_dim = E.shape

    D0 = torch.tensor(
        [
            [1 - nu, nu, 0.0],
            [nu, 1 - nu, 0.0],
            [0.0, 0.0, (1 - 2 * nu) / 2.0],
        ],
        device=device,
    )

    return coeff[..., None, None] * D0 # Shape: [B, N_ELEMS, 3, 3]

# ==========================================
# 2. FULLY BATCHED FEM SOLVER (NO LOOPS)
# ==========================================
def shape_function_derivatives(xi, eta):
    dN_dxi = 0.25 * torch.stack([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * torch.stack([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return dN_dxi, dN_deta

# Precompute element coordinates and Gauss integration details
elem_coords = nodes[elements] # Shape: [N_ELEMS, 4, 2]
g = 1.0 / np.sqrt(3.0)
gauss_pts = torch.tensor([-g, g], device=device, dtype=torch.float32)

# Precompute centers of elements for material properties evaluation
elem_centers_x = elem_coords[:, :, 0].mean(dim=1)
elem_centers_y = elem_coords[:, :, 1].mean(dim=1)

# Precalculate Batched Element Stiffness matrices template across Gauss Points
B_elements_gp = []
detJ_gp = []

for xi in gauss_pts:
    for eta in gauss_pts:
        dN_dxi, dN_deta = shape_function_derivatives(xi, eta) # Shape: [4]
        
        # J = dN_dxi/eta @ elem_coords -> [N_ELEMS, 2, 2]
        J = torch.zeros(N_ELEMS, 2, 2, device=device)
        for i in range(4):
            J[:, 0, 0] += dN_dxi[i] * elem_coords[:, i, 0]
            J[:, 0, 1] += dN_dxi[i] * elem_coords[:, i, 1]
            J[:, 1, 0] += dN_deta[i] * elem_coords[:, i, 0]
            J[:, 1, 1] += dN_deta[i] * elem_coords[:, i, 1]
            
        detJ = torch.det(J) # [N_ELEMS]
        invJ = torch.inverse(J) # [N_ELEMS, 2, 2]
        
        # Compute B matrices per element
        B = torch.zeros(N_ELEMS, 3, 8, device=device)
        for i in range(4):
            grad = torch.stack([dN_dxi[i].expand(N_ELEMS), dN_deta[i].expand(N_ELEMS)], dim=-1).unsqueeze(1) # [N_ELEMS, 1, 2]
            dN_i = torch.bmm(grad, invJ).squeeze(1) # [N_ELEMS, 2]
            
            B[:, 0, 2 * i] = dN_i[:, 0]
            B[:, 1, 2 * i + 1] = dN_i[:, 1]
            B[:, 2, 2 * i] = dN_i[:, 1]
            B[:, 2, 2 * i + 1] = dN_i[:, 0]
            
        B_elements_gp.append(B)
        detJ_gp.append(detJ)

# Precompute DOFs mapping
dofs = torch.zeros((N_ELEMS, 8), dtype=torch.long, device=device)
for i in range(4):
    dofs[:, 2 * i] = 2 * elements[:, i]
    dofs[:, 2 * i + 1] = 2 * elements[:, i] + 1

# Identify boundary conditions
fixed_nodes = torch.where(torch.abs(nodes[:, 0]) < 1e-6)[0]
fixed_dofs = torch.cat([2 * fixed_nodes, 2 * fixed_nodes + 1])
traction_nodes = torch.where(torch.abs(nodes[:, 0] - LX) < 1e-6)[0]


B_avg = torch.stack(
    [
        torch.stack(
            [
                B_elements_gp[gp][elem]
                for gp in range(4)
            ],
            dim=0
        ).mean(dim=0)
        for elem in range(N_ELEMS)
    ],
    dim=0
).to(device)
# print(B_avg.shape)
def fem_solver_batched(raw_E_nodes):
    # theta shape: [B, 6]
    B_dim = raw_E_nodes.shape[0]
    ndofs = 2 * N_NODES
    
    # 1. Compute Batched Material Fields
    E_nodes = E_MIN + (E_MAX - E_MIN) * torch.sigmoid(raw_E_nodes)
    E_elem = nodal_E_to_element_E(E_nodes)
    D = elasticity_matrix_batched(E_elem)# [B, N_ELEMS, 3, 3]
    
    # 2. Integrate Element Stiffness Ke -> Shape: [B, N_ELEMS, 8, 8]
    ke_batched = torch.zeros(B_dim, N_ELEMS, 8, 8, device=device)
    for B_gp, detJ in zip(B_elements_gp, detJ_gp):
        # B_gp: [N_ELEMS, 3, 8], D: [B, N_ELEMS, 3, 3]
        # Perform batched matrix multiplication across all batches and elements
        tmp = torch.matmul(B_gp.transpose(-1, -2).unsqueeze(0), D) # [B, N_ELEMS, 8, 3]
        ke_gp = torch.matmul(tmp, B_gp.unsqueeze(0)) * detJ.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        ke_batched += ke_gp

    # 3. Assemble Global Stiffness Matrix K
    K = torch.zeros(B_dim, ndofs, ndofs, device=device)
    # Scatter element stiffness matrices to global K
    rows = dofs.unsqueeze(-1).expand(-1, -1, 8)
    cols = dofs.unsqueeze(1).expand(-1, 8, -1)
    
    K.index_put_((torch.arange(B_dim, device=device).view(-1, 1, 1, 1), 
                  rows.unsqueeze(0), cols.unsqueeze(0)), ke_batched, accumulate=True)
    
    # 4. Assemble Force Vector F
    F = torch.zeros(B_dim, ndofs, device=device)
    traction = 50e6
    F[:, 2 * traction_nodes] += traction # Traction force
    
    # 5. Apply Boundary Conditions
    K[:, fixed_dofs, :] = 0.0
    K[:, :, fixed_dofs] = 0.0
    K[:, fixed_dofs, fixed_dofs] = 1.0
    F[:, fixed_dofs] = 0.0
    
    # Regularization for numerical stability
    # K += 1e6 * torch.eye(ndofs, device=device).unsqueeze(0)
    
    # Batched Solver
    U = torch.linalg.solve(K, F)
    return U

def compute_strains_batched(U):

    Ue = U[:, dofs]     # [B,N_ELEMS,8]

    strain = torch.einsum(
        "eij,bej->bei",
        B_avg,
        Ue
    )

    return strain

# ==========================================
# 3. PIPELINE SETUP & MODEL ARCHITECTURE
# ==========================================
N_STRAIN_SENSORS = 40
N_SENSORS = 3*N_STRAIN_SENSORS
strain_sensor_elems = torch.randperm(
    N_ELEMS,
    device=device
)[:N_STRAIN_SENSORS]


# sensor_indices = torch.randperm(2 * N_NODES, device=device)[:N_SENSORS]

def measure_batched(U):

    strain = compute_strains_batched(U)

    strain_obs = strain[
        :,
        strain_sensor_elems,
        :
    ]

    return strain_obs.reshape(
        U.shape[0],
        -1
    )

# --- cINN Architecture ---
SUMMARY_DIM = 256
THETA_DIM = N_NODES
N_COUPLINGS = 16

class SummaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_SENSORS, 1024), nn.Tanh(),
            nn.Linear(1024, 1024), nn.Tanh(),
            # nn.Linear(256, 256), nn.Tanh(),
            # nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(1024, SUMMARY_DIM)
        )
    def forward(self, y): return self.net(y)

class CouplingNet(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, 1024), nn.Tanh(),
            nn.Linear(1024, 1024), nn.Tanh(),
            # nn.Linear(1024, 1024), nn.Tanh(),
            # nn.Linear(1024, 1024), nn.Tanh(),
            nn.Linear(1024, out_dim)
        )
    def forward(self, x, h): return self.net(torch.cat([x, h], dim=-1))

class AffineCoupling(nn.Module):
    def __init__(self):
        super().__init__()
        self.split = THETA_DIM // 2
        self.s_net = CouplingNet(self.split, SUMMARY_DIM, THETA_DIM - self.split)
        self.t_net = CouplingNet(self.split, SUMMARY_DIM, THETA_DIM - self.split)
    def forward(self, x, h):
        x1, x2 = x[:, :self.split], x[:, self.split:]
        s = 0.1 * torch.tanh(self.s_net(x1, h))
        t = self.t_net(x1, h)
        y2 = x2 * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return torch.cat([x1, y2], dim=-1), log_det
    def inverse(self, y, h):
        y1, y2 = y[:, :self.split], y[:, self.split:]
        s = 0.1 * torch.tanh(self.s_net(y1, h))
        t = self.t_net(y1, h)
        x2 = (y2 - t) * torch.exp(-s)
        inv_log_det = -torch.sum(s, dim=1)
        return torch.cat([y1, x2], dim=-1), inv_log_det

class Permutation(nn.Module):
    def __init__(self):
        super().__init__()
        perm = torch.randperm(THETA_DIM)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))
    def forward(self, x): return x[:, self.perm]
    def inverse(self, x): return x[:, self.inv_perm]

class ConditionalINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = SummaryNet()
        self.couplings = nn.ModuleList([AffineCoupling() for _ in range(N_COUPLINGS)])
        self.perms = nn.ModuleList([Permutation() for _ in range(N_COUPLINGS)])

    def forward(self, theta, y):
        h = self.summary(y)
        z = theta
        log_det = torch.zeros(theta.shape[0], device=theta.device)
        for coupling, perm in zip(self.couplings, self.perms):
            z = perm(z)
            z, ld = coupling(z, h)
            log_det += ld
        return z, log_det

    def sample(self, z, y):
        h = self.summary(y)
        u = z
        inv_log_det = torch.zeros(z.shape[0], device=z.device)
        for coupling, perm in reversed(list(zip(self.couplings, self.perms))):
            u, ld = coupling.inverse(u, h)
            u = perm.inverse(u)
            inv_log_det += ld
        return u, inv_log_det

def log_q(model, theta, y):
    z, log_det = model(theta, y)
    log_pz = -0.5 * torch.sum(z**2, dim=1)
    return log_pz + log_det

# ==========================================
# 4. DATA GENERATION & LOSS PIPELINE
# ==========================================
# def sample_theta(batch_size):

#     E = (
#         E_MIN
#         + (E_MAX - E_MIN)
#         * torch.rand(
#             batch_size,
#             N_NODES,
#             device=device
#         )
#     )

#     s = (E - E_MIN) / (E_MAX - E_MIN)

#     raw_E = torch.log(
#         s / (1.0 - s)
#     )

#     return raw_E

# N_DATA = 10
# print("Generating Offline Dataset in Batches...")
# # Processing dataset generation in batches instead of single loops
# GEN_BATCH_SIZE = 10
# theta_dataset = []
# y_dataset = []

# with torch.no_grad():
#     for i in range(0, N_DATA, GEN_BATCH_SIZE):
#         batch_theta = sample_theta(GEN_BATCH_SIZE)
#         batch_U = fem_solver_batched(batch_theta)
#         batch_y = measure_batched(batch_U)
#         batch_y += 0.000001 * torch.randn_like(batch_y)
        
#         theta_dataset.append(batch_theta.cpu())
#         y_dataset.append(batch_y.cpu())
#         print(f"Generated {i + GEN_BATCH_SIZE}/{N_DATA}")

# theta_dataset = torch.cat(theta_dataset, dim=0)
# y_dataset = torch.cat(y_dataset, dim=0)

# y_mean = y_dataset.mean(dim=0)
# y_std = y_dataset.std(dim=0) + 1e-6
# def normalize_y(y): return (y - y_mean.to(y.device)) / y_std.to(y.device)

# # --- Optimized Misfit Energy Function ---
# LAMBDA_DATA = 5.0e11
# LAMBDA_PRIOR = 1.0
# LAMBDA_ENT = 0.1
# TAU = 0.5
# OUTER_JKO_STEPS = 5
# INNER_EPOCHS = 5000
# BATCH_SIZE = 100

# def posterior_energy_batched(theta_batch, y_batch):
#     # theta_batch: [B, THETA_DIM], y_batch: [B, N_SENSORS]
#     U_pred = fem_solver_batched(theta_batch)
#     y_pred = measure_batched(U_pred)
    
#     misfit = 0.5 * torch.mean((y_pred - y_batch) ** 2, dim=1)
#     E = (
#         E_MIN
#         + (E_MAX - E_MIN)
#         * torch.sigmoid(theta_batch)
#     )
    
#     s = (E - E_MIN) / (E_MAX - E_MIN)
    
#     alpha = 0.5
    
#     log_prior = (
#         (alpha - 1.0) * torch.log(s)
#         + (alpha - 1.0) * torch.log(1.0 - s)
#     )
    
#     prior = -torch.sum(log_prior, dim=1)
#     # prior = 0.5 * torch.sum(theta_batch ** 2, dim=1)
    
#     return LAMBDA_DATA * misfit + LAMBDA_PRIOR * prior


# ==========================================
# 4. DATA GENERATION & LOSS PIPELINE
# ==========================================

# Gaussian prior parameters (in Pa)
E_MEAN = 200e9
E_STD = 30e9


# ------------------------------------------
# Sample Gaussian prior on physical E
# ------------------------------------------

def sample_theta(batch_size):

    E = torch.empty(
        batch_size,
        N_NODES,
        device=device
    )

    mask = torch.ones_like(E, dtype=torch.bool)

    # Rejection sampling for bounded Gaussian
    while mask.any():

        candidates = (
            E_MEAN
            + E_STD * torch.randn(
                batch_size,
                N_NODES,
                device=device
            )
        )

        valid = (
            (candidates >= E_MIN)
            & (candidates <= E_MAX)
        )

        accept = mask & valid

        E[accept] = candidates[accept]
        mask[accept] = False

    # Physical E -> normalized s -> unconstrained theta
    s = (E - E_MIN) / (E_MAX - E_MIN)

    eps = 1e-6
    s = torch.clamp(s, eps, 1.0 - eps)

    raw_E = torch.log(s) - torch.log1p(-s)

    return raw_E


# ------------------------------------------
# Offline dataset generation
# ------------------------------------------

N_DATA = 10
GEN_BATCH_SIZE = 10

print("Generating Offline Dataset in Batches...")

theta_dataset = []
y_dataset = []

with torch.no_grad():

    for i in range(0, N_DATA, GEN_BATCH_SIZE):

        current_batch = min(
            GEN_BATCH_SIZE,
            N_DATA - i
        )

        batch_theta = sample_theta(current_batch)

        batch_U = fem_solver_batched(batch_theta)

        batch_y = measure_batched(batch_U)

        batch_y += 0.000001 * torch.randn_like(batch_y)

        theta_dataset.append(batch_theta.cpu())
        y_dataset.append(batch_y.cpu())

        print(f"Generated {i + current_batch}/{N_DATA}")


theta_dataset = torch.cat(theta_dataset, dim=0)
y_dataset = torch.cat(y_dataset, dim=0)


# ------------------------------------------
# Observation normalization
# ------------------------------------------

y_mean = y_dataset.mean(dim=0)
y_std = y_dataset.std(dim=0) + 1e-6

def normalize_y(y):
    return (
        y - y_mean.to(y.device)
    ) / y_std.to(y.device)


# ------------------------------------------
# JKO parameters
# ------------------------------------------

LAMBDA_DATA = 5.0e11
LAMBDA_PRIOR = 1.0
LAMBDA_ENT = 0.1

TAU = 0.5

OUTER_JKO_STEPS = 30
INNER_EPOCHS = 5000
BATCH_SIZE = 100


# ==========================================
# POSTERIOR ENERGY WITH GAUSSIAN PRIOR
# ==========================================

NOISE_STD = 1e-6

def posterior_energy_batched(theta_batch, y_batch):

    # FEM prediction
    U_pred = fem_solver_batched(theta_batch)
    y_pred = measure_batched(U_pred)

    # Gaussian negative log-likelihood
    data_energy = 0.5 * torch.sum(
        ((y_pred - y_batch) / NOISE_STD)**2,
        dim=1
    )

    # Physical Young's modulus
    E = E_MIN + (E_MAX - E_MIN) * torch.sigmoid(theta_batch)

    # Gaussian negative log-prior
    prior = 0.5 * torch.sum(
        ((E - E_MEAN) / E_STD)**2,
        dim=1
    )

    # Jacobian of the sigmoid transformation
    log_jacobian = (
        np.log(E_MAX - E_MIN)
        + torch.nn.functional.logsigmoid(theta_batch)
        + torch.nn.functional.logsigmoid(-theta_batch)
    )

    prior -= torch.sum(log_jacobian, dim=1)

    return data_energy + prior


# ==========================================
# 5. JKO TRAINING LOOP
# ==========================================


# In[ ]:


model = ConditionalINN().to(device)
from snrGD import SNRGatedDescent as SNRGD
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = SNRGD(model.parameters(), m=2, lr= 1.0)
sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.005)
reference_model = copy.deepcopy(model)

losses_JKO = []
losses_V = []
for k in range(OUTER_JKO_STEPS):

    print(f"\n========== JKO STEP {k} ==========")

    # --------------------------------------------------
    # Reference distribution q_k
    # --------------------------------------------------

    reference_model.load_state_dict(
        model.state_dict()
    )

    reference_model.eval()

    for p in reference_model.parameters():
        p.requires_grad = False

    # --------------------------------------------------
    # Track best inner iterate
    # --------------------------------------------------

    best_loss = float("inf")
    best_state = None

    # --------------------------------------------------
    # Inner optimization
    # --------------------------------------------------

    for epoch in range(INNER_EPOCHS):

        # ----------------------------------------------
        # minibatch
        # ----------------------------------------------

        idx = torch.randint(
            0,
            len(y_dataset),
            (BATCH_SIZE,),
            # device=device
        )

        y_batch = y_dataset[idx].to(device)
        y_cond = normalize_y(y_batch)

        # ----------------------------------------------
        # Current samples
        # ----------------------------------------------

        z = torch.randn(
            BATCH_SIZE,
            THETA_DIM,
            device=device
        )

        raw_E_cur, inv_log_det = model.sample(
            z,
            y_cond
        )

        # ----------------------------------------------
        # log q(theta|y)
        # ----------------------------------------------

        log_pz = -0.5 * torch.sum(
            z**2,
            dim=1
        )

        logq = log_pz - inv_log_det

        # ----------------------------------------------
        # Reference samples
        # ----------------------------------------------

        with torch.no_grad():

            # z_ref = torch.randn(
            #     BATCH_SIZE,
            #     THETA_DIM,
            #     device=device
            # )

            raw_E_ref, _ = reference_model.sample(
                z,
                y_cond
            )

        # ----------------------------------------------
        # Energy
        # ----------------------------------------------

        V = posterior_energy_batched(
            raw_E_cur,
            y_batch
        )

        kl_term = torch.mean(
            V + logq
        )

        prox_term = sinkhorn_loss(
            raw_E_cur,
            raw_E_ref
        )

        loss = (
            kl_term
            + (1.0 / (2.0 * TAU))
            * prox_term
        )

        # ----------------------------------------------
        # Gradient step
        # ----------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        # torch.cuda.synchronize()
        
        # print("Allocated before step:",
        #       torch.cuda.memory_allocated() / 1024**3)
        
        # print("Reserved before step:",
        #       torch.cuda.memory_reserved() / 1024**3)
        
        # print("Peak memory:",
        #       torch.cuda.max_memory_allocated() / 1024**3)
        
        optimizer.step()

        # optimizer.step()

        # ----------------------------------------------
        # Save best iterate
        # ----------------------------------------------

        cur_loss = loss.item()
        losses_JKO.append(loss.item())
        losses_V.append(V.mean().item())

        if cur_loss < best_loss:

            best_loss = cur_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        # ----------------------------------------------
        # Logging
        # ----------------------------------------------

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch:4d} | "
                f"Loss={loss.item():.4f} | "
                f"KL={kl_term.item():.4f} | "
                f"V={V.mean().item():.4f} | "
                f"Ent={logq.mean().item():.4f} | "
                f"Sinkhorn={prox_term.item():.4f}"
            )

    # --------------------------------------------------
    # Restore best inner iterate
    # --------------------------------------------------

    model.load_state_dict(best_state)

    print(
        f"Best loss at JKO step {k}: "
        f"{best_loss:.4f}"
    )


# In[32]:


# import matplotlib.pyplot as plt

# # Extracted data from the logs
# jko_steps = list(range(21))
# best_losses = [ 28298.4102,
#     1013.9408, 741.6211, 669.6744, 635.8552, 605.5137,
#     593.3389,  576.2623, 575.6964, 552.0284, 545.4344,
#     537.9429,  535.1328, 523.5163, 514.9293, 510.5664,
#     506.7003,  498.9523, 495.8867, 483.4910, 483.4911
# ]

# # Create the figure
# # plt.figure(figsize=(8, 6))

# # Plot the trajectory line and individual markers
# plt.plot(jko_steps, best_losses, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6, label='Best Loss')

# # Graph customizations
# # plt.title('Best Loss Trajectory Across JKO Steps', fontsize=14, fontweight='bold', pad=15)
# plt.xlabel('JKO Step', fontsize=12)
# plt.ylabel('loss', fontsize=12)
# plt.yscale('log')
# plt.xticks(jko_steps)  # Forces integer ticks for all steps
# plt.grid(True, linestyle='--', alpha=0.6)

# # Annotate the final step value
# # plt.annotate(f'{best_losses[-1]:.4f}', 
# #              xy=(jko_steps[-1], best_losses[-1]), 
# #              xytext=(jko_steps[-1]-1.5, best_losses[-1]+40),
# #              arrowprops=dict(arrowstyle="->", color='black', lw=1),
# #              fontsize=10, fontweight='bold')

# plt.tight_layout()
# plt.show()


# In[3]:


# ============================================================
# 6. EVALUATION ON A COMPLETELY NEW OBSERVATION
# ============================================================

import matplotlib.pyplot as plt

model.eval()

N_POST = 2000
N_PRIOR = 5000
POST_BATCH_SIZE = 128
N_RANDOM_NODES = 8

# ------------------------------------------------------------
# 1. Generate a NEW independent test material field
# ------------------------------------------------------------

with torch.no_grad():

    # Fresh draw from the Gaussian prior
    raw_E_test = sample_theta(1)

    # Corresponding physical Young's modulus
    E_true = (
        E_MIN
        + (E_MAX - E_MIN) * torch.sigmoid(raw_E_test)
    )

    # Generate displacement using the true material field
    U_test = fem_solver_batched(raw_E_test)

    # Generate strain measurements
    y_test_clean = measure_batched(U_test)

    # Add measurement noise, consistent with training data
    y_test = y_test_clean + 1e-6 * torch.randn_like(y_test_clean)

    # Normalize using TRAINING dataset statistics
    y_test_normalized = normalize_y(y_test)

print("New test observation shape:", y_test.shape)
print("New true material field shape:", E_true.shape)


# ------------------------------------------------------------
# 2. Generate posterior samples conditioned on new observation
# ------------------------------------------------------------

posterior_samples = []

with torch.no_grad():

    for start in range(0, N_POST, POST_BATCH_SIZE):

        current_batch = min(
            POST_BATCH_SIZE,
            N_POST - start
        )

        z = torch.randn(
            current_batch,
            THETA_DIM,
            device=device
        )

        y_condition = y_test_normalized.repeat(
            current_batch, 1
        )

        raw_E_post, _ = model.sample(
            z,
            y_condition
        )

        # Transform into physical Young's modulus
        E_post = (
            E_MIN
            + (E_MAX - E_MIN) * torch.sigmoid(raw_E_post)
        )

        posterior_samples.append(E_post.cpu())


posterior_samples = torch.cat(
    posterior_samples,
    dim=0
)

# Posterior mean and standard deviation
E_post_mean = posterior_samples.mean(dim=0)
E_post_std = posterior_samples.std(dim=0)

# True field in CPU representation
E_true_cpu = E_true[0].cpu()

# Convert to GPa for plotting
E_true_GPa = E_true_cpu.numpy() / 1e9
E_post_mean_GPa = E_post_mean.numpy() / 1e9
E_post_std_GPa = E_post_std.numpy() / 1e9


# ============================================================
# 3. COMPUTE SPATIAL ERRORS
# ============================================================

error = E_post_mean - E_true_cpu

absolute_error = torch.abs(error)

# Reshape into structured grid
error_grid = error.reshape(NY, NX)
true_grid = E_true_cpu.reshape(NY, NX)

# Trapezoidal integration weights
weights = torch.ones(NY, NX)

weights[0, :] *= 0.5
weights[-1, :] *= 0.5
weights[:, 0] *= 0.5
weights[:, -1] *= 0.5

# Spatial L2 error
L2_error = torch.sqrt(
    torch.sum(weights * error_grid**2) * dx * dy
)

# Relative L2 error
L2_true = torch.sqrt(
    torch.sum(weights * true_grid**2) * dx * dy
)

relative_L2 = L2_error / (L2_true + 1e-12)

RMSE = torch.sqrt(torch.mean(error**2))

print("\n========== NEW OBSERVATION: INFERENCE ERRORS ==========")
print(f"L2 error          : {L2_error.item()/1e9:.6f} GPa")
print(f"Relative L2 error : {relative_L2.item():.6f}")
print(f"Relative L2 (%)   : {100*relative_L2.item():.4f}%")
print(f"RMSE              : {RMSE.item()/1e9:.6f} GPa")


# ============================================================
# PLOT 1: TRUE E, POSTERIOR MEAN E AND ABSOLUTE ERROR
# ============================================================

X = x_grid.cpu().numpy()
Y = y_grid.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

im0 = axes[0].contourf(
    X, Y,
    E_true_GPa.reshape(NY, NX),
    levels=30,
    cmap="viridis"
)

axes[0].set_title("True Young's modulus")
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$y$")
fig.colorbar(im0, ax=axes[0], label="$E$ (GPa)")


im1 = axes[1].contourf(
    X, Y,
    E_post_mean_GPa.reshape(NY, NX),
    levels=30,
    cmap="viridis"
)

axes[1].set_title("Posterior mean Young's modulus")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$y$")
fig.colorbar(im1, ax=axes[1], label="$E$ (GPa)")


im2 = axes[2].contourf(
    X, Y,
    absolute_error.numpy().reshape(NY, NX) / 1e9,
    levels=30,
    cmap="inferno"
)

axes[2].set_title(
    f"Absolute error\nRelative $L_2$ = {relative_L2.item():.4f}"
)

axes[2].set_xlabel("$x$")
axes[2].set_ylabel("$y$")

fig.colorbar(im2, ax=axes[2], label="Error (GPa)")

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 2: DISPLACEMENT FIELDS
# ============================================================

# Posterior mean physical E -> unconstrained raw E
s_mean = (E_post_mean.to(device) - E_MIN) / (E_MAX - E_MIN)

eps = 1e-6
s_mean = torch.clamp(s_mean, eps, 1.0 - eps)

raw_E_post_mean = (
    torch.log(s_mean) - torch.log1p(-s_mean)
).unsqueeze(0)

with torch.no_grad():

    U_post_mean = fem_solver_batched(raw_E_post_mean)

    # U_test was calculated using the true material field
    U_true = U_test


# Extract displacement components
ux_true = U_true[0, 0::2].reshape(NY, NX).cpu().numpy()
uy_true = U_true[0, 1::2].reshape(NY, NX).cpu().numpy()

ux_post = U_post_mean[0, 0::2].reshape(NY, NX).cpu().numpy()
uy_post = U_post_mean[0, 1::2].reshape(NY, NX).cpu().numpy()

# Convert m to mm
ux_true *= 1e3
uy_true *= 1e3
ux_post *= 1e3
uy_post *= 1e3

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

fields = [
    (ux_true, "$u_x$ at true $E$"),
    (ux_post, "$u_x$ at posterior mean $E$"),
    (uy_true, "$u_y$ at true $E$"),
    (uy_post, "$u_y$ at posterior mean $E$")
]

for ax, (field, title) in zip(axes.flatten(), fields):

    im = ax.contourf(
        X, Y, field,
        levels=30,
        cmap="coolwarm"
    )

    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.colorbar(im, ax=ax, label="Displacement (mm)")

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 3: POSTERIOR MEAN VS TRUE E
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    E_true_GPa,
    E_post_mean_GPa,
    alpha=0.65,
    s=25,
    edgecolors="none"
)

lower = min(
    E_true_GPa.min(),
    E_post_mean_GPa.min()
)

upper = max(
    E_true_GPa.max(),
    E_post_mean_GPa.max()
)

ax.plot(
    [lower, upper],
    [lower, upper],
    "r--",
    linewidth=2,
    label="Perfect agreement"
)

ax.set_xlabel("True $E$ (GPa)")
ax.set_ylabel("Posterior mean $E$ (GPa)")

ax.set_title("Posterior mean versus true Young's modulus")

ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 4. GENERATE PRIOR SAMPLES
# ============================================================

with torch.no_grad():

    raw_E_prior = sample_theta(N_PRIOR)

    E_prior = (
        E_MIN
        + (E_MAX - E_MIN) * torch.sigmoid(raw_E_prior)
    )

E_prior_GPa = E_prior.cpu().numpy() / 1e9
E_post_GPa = posterior_samples.numpy() / 1e9


# ============================================================
# PLOT 4: PRIOR AND POSTERIOR AT 8 RANDOM NODES
# ============================================================

torch.manual_seed(123)

random_nodes = torch.randperm(N_NODES)[:N_RANDOM_NODES].tolist()

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

axes = axes.flatten()

for ax, node_id in zip(axes, random_nodes):

    ax.hist(
        E_prior_GPa[:, node_id],
        bins=35,
        density=True,
        alpha=0.45,
        color="steelblue",
        label="Prior"
    )

    ax.hist(
        E_post_GPa[:, node_id],
        bins=35,
        density=True,
        alpha=0.55,
        color="orange",
        label="Posterior"
    )

    ax.axvline(
        E_true_GPa[node_id],
        color="red",
        linestyle="--",
        linewidth=2,
        label="True $E$"
    )

    x_coord = nodes[node_id, 0].item()
    y_coord = nodes[node_id, 1].item()

    ax.set_title(
        f"Node {node_id}\n"
        f"$(x,y)=({x_coord:.2f},{y_coord:.2f})$"
    )

    ax.set_xlabel("$E$ (GPa)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.02)
)

fig.suptitle(
    "Prior and posterior distributions at selected nodes",
    y=1.07
)

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 5: PRIOR AND POSTERIOR OVER ALL NODES
# ============================================================

prior_all = E_prior_GPa.flatten()
posterior_all = E_post_GPa.flatten()

true_all = E_true_GPa.flatten()

bins = np.linspace(
    E_MIN / 1e9,
    E_MAX / 1e9,
    60
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    prior_all,
    bins=bins,
    density=True,
    alpha=0.4,
    color="steelblue",
    label="Prior"
)

ax.hist(
    posterior_all,
    bins=bins,
    density=True,
    alpha=0.5,
    color="orange",
    label="Posterior"
)

ax.hist(
    true_all,
    bins=bins,
    density=True,
    histtype="step",
    linestyle="--",
    linewidth=2.2,
    color="red",
    label="True field"
)

ax.set_xlabel("Young's modulus $E$ (GPa)")
ax.set_ylabel("Probability density")

ax.set_title(
    "Prior and posterior distributions over all nodes"
)

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== NEW OBSERVATION: SUMMARY ==========")

print(
    f"True spatial mean E      : "
    f"{E_true.mean().item()/1e9:.4f} GPa"
)

print(
    f"Posterior spatial mean E : "
    f"{E_post_mean.mean().item()/1e9:.4f} GPa"
)

print(
    f"True spatial std E       : "
    f"{E_true.std().item()/1e9:.4f} GPa"
)

print(
    f"Posterior spatial std E  : "
    f"{E_post_mean.std().item()/1e9:.4f} GPa"
)

print(
    f"Mean posterior std       : "
    f"{E_post_std.mean().item()/1e9:.4f} GPa"
)

print(
    f"Relative L2 error        : "
    f"{relative_L2.item():.6f}"
)


# In[4]:


# =====================================
# COMMON TEST CASE
# =====================================
def sample_theta_range(
    batch_size,
    smin=0.0,
    smax=1.0
):

    s = smin + (smax-smin) * torch.rand(
        batch_size,
        N_NODES,
        device=device
    )

    raw_E = torch.log(
        s/(1-s)
    )

    return raw_E
    
torch.manual_seed(123)

with torch.no_grad():

    raw_E_true = sample_theta_range(1, 0.0,1.0)

    U_true = fem_solver_batched(raw_E_true)

    y_obs = measure_batched(U_true)

    y_obs += 1e-5 * torch.randn_like(y_obs)

NSAMPLES = 5000

z_test = torch.randn(
    NSAMPLES,
    THETA_DIM,
    device=device
)


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import torch

# ============================================================
# 1. RANDOM TRUE FIELD
# ============================================================

with torch.no_grad():

    # raw_E_true = sample_theta(1)[0]

    E_true = (
        E_MIN
        + (E_MAX - E_MIN)
        * torch.sigmoid(raw_E_true[0])
    )

# ============================================================
# 3. POSTERIOR SAMPLING
# ============================================================

NSAMPLES = 5000

with torch.no_grad():

    y_cond = normalize_y(y_obs)

    y_cond = y_cond.repeat(
        NSAMPLES,
        1
    )

    z = torch.randn(
        NSAMPLES,
        THETA_DIM,
        device=device
    )

    raw_samples, _ = model.sample(
        z,
        y_cond
    )

    E_samples = (
        E_MIN
        + (E_MAX - E_MIN)
        * torch.sigmoid(raw_samples)
    )

# ============================================================
# 4. POSTERIOR MEAN FIELD
# ============================================================

with torch.no_grad():

    raw_E_mean = raw_samples.mean(0)
    
    E_mean = E_samples.mean(0)
    
    s_mean = (E_mean - E_MIN) / (E_MAX - E_MIN)
    
    raw_E_from_Emean = torch.log(
        s_mean / (1.0 - s_mean)
    )
    
    U_post = fem_solver_batched(
        raw_E_from_Emean.unsqueeze(0)
    )[0]

# ============================================================
# 5. RESHAPE E FIELDS
# ============================================================

E_true_grid = (
    E_true.cpu()
    .numpy()
    .reshape(NY, NX)
)

E_mean_grid = (
    E_mean.cpu()
    .numpy()
    .reshape(NY, NX)
)

E_err_grid = np.abs(
    E_mean_grid - E_true_grid
)

# ============================================================
# 6. RESHAPE DISPLACEMENTS
# ============================================================
U_true_flat = U_true[0]
Ux_true = (
    U_true_flat[0::2]
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Uy_true = (
    U_true_flat[1::2]
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Ux_post = (
    U_post[0::2]
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Uy_post = (
    U_post[1::2]
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Ux_err = np.abs(
    Ux_post - Ux_true
)

Uy_err = np.abs(
    Uy_post - Uy_true
)

# ============================================================
# 7. E FIELD PLOTS
# ============================================================

fig, ax = plt.subplots(
    1,
    3,
    figsize=(18,5)
)

emin = min(
    E_true_grid.min(),
    E_mean_grid.min()
)

emax = max(
    E_true_grid.max(),
    E_mean_grid.max()
)

im = ax[0].imshow(
    E_true_grid/1e9,
    origin="lower",
    vmin=emin/1e9,
    vmax=emax/1e9
)
ax[0].set_title("True E (GPa)")
fig.colorbar(im, ax=ax[0])

im = ax[1].imshow(
    E_mean_grid/1e9,
    origin="lower",
    vmin=emin/1e9,
    vmax=emax/1e9
)
ax[1].set_title("Posterior Mean E (GPa)")
fig.colorbar(im, ax=ax[1])

im = ax[2].imshow(
    E_err_grid/1e9,
    origin="lower"
)
ax[2].set_title("|E Error| (GPa)")
fig.colorbar(im, ax=ax[2])

plt.tight_layout()
plt.show()

# ============================================================
# 8. Ux / Uy COMPARISON
# ============================================================

fig, ax = plt.subplots(
    2,
    3,
    figsize=(18,10)
)

# -----------------------------
# Ux
# -----------------------------

uxmin = min(
    Ux_true.min(),
    Ux_post.min()
)

uxmax = max(
    Ux_true.max(),
    Ux_post.max()
)

im = ax[0,0].imshow(
    Ux_true,
    origin="lower",
    vmin=uxmin,
    vmax=uxmax
)
ax[0,0].set_title("True Ux")
fig.colorbar(im, ax=ax[0,0])

im = ax[0,1].imshow(
    Ux_post,
    origin="lower",
    vmin=uxmin,
    vmax=uxmax
)
ax[0,1].set_title("Posterior Mean Ux")
fig.colorbar(im, ax=ax[0,1])

im = ax[0,2].imshow(
    Ux_err,
    origin="lower"
)
ax[0,2].set_title("|Ux Error|")
fig.colorbar(im, ax=ax[0,2])

# -----------------------------
# Uy
# -----------------------------

uymin = min(
    Uy_true.min(),
    Uy_post.min()
)

uymax = max(
    Uy_true.max(),
    Uy_post.max()
)

im = ax[1,0].imshow(
    Uy_true,
    origin="lower",
    vmin=uymin,
    vmax=uymax
)
ax[1,0].set_title("True Uy")
fig.colorbar(im, ax=ax[1,0])

im = ax[1,1].imshow(
    Uy_post,
    origin="lower",
    vmin=uymin,
    vmax=uymax
)
ax[1,1].set_title("Posterior Mean Uy")
fig.colorbar(im, ax=ax[1,1])

im = ax[1,2].imshow(
    Uy_err,
    origin="lower"
)
ax[1,2].set_title("|Uy Error|")
fig.colorbar(im, ax=ax[1,2])

plt.tight_layout()
plt.show()

# ============================================================
# 9. NUMERICAL ERROR METRICS
# ============================================================

E_rel = (
    np.linalg.norm(E_mean_grid - E_true_grid)
    /
    np.linalg.norm(E_true_grid)
)

Ux_rel = (
    np.linalg.norm(Ux_post - Ux_true)
    /
    np.linalg.norm(Ux_true)
)

Uy_rel = (
    np.linalg.norm(Uy_post - Uy_true)
    /
    np.linalg.norm(Uy_true)
)

print("\nRelative Errors")
print("----------------")
print("E  :", E_rel)
print("Ux :", Ux_rel)
print("Uy :", Uy_rel)

print("\nMaximum Absolute Errors")
print("-----------------------")
print("E  :", E_err_grid.max()/1e9, "GPa")
print("Ux :", Ux_err.max())
print("Uy :", Uy_err.max())


# In[6]:


print("\nAvg Absolute Errors")
print("-----------------------")
print("E  :", E_err_grid.mean()/1e9, "GPa")


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import torch

# ============================================================
# 1. RANDOM TRUE FIELD
# ============================================================

with torch.no_grad():

    # raw_E_true = sample_theta(1)[0]

    E_true = (
        E_MIN
        + (E_MAX - E_MIN)
        * torch.sigmoid(raw_E_true[0])
    )

# ============================================================
# 3. POSTERIOR SAMPLING
# ============================================================

NSAMPLES = 5000

with torch.no_grad():

    y_cond = normalize_y(y_obs)

    y_cond = y_cond.repeat(
        NSAMPLES,
        1
    )

    z = torch.randn(
        NSAMPLES,
        THETA_DIM,
        device=device
    )

    raw_samples, _ = model.sample(
        z,
        y_cond
    )

    E_samples = (
        E_MIN
        + (E_MAX - E_MIN)
        * torch.sigmoid(raw_samples)
    )

# ============================================================
# 4. POSTERIOR MEAN FIELD
# ============================================================

with torch.no_grad():

    raw_E_mean = raw_samples.mean(0)
    
    E_mean = E_samples.mean(0)
    
    s_mean = (E_mean - E_MIN) / (E_MAX - E_MIN)
    
    raw_E_from_Emean = torch.log(
        s_mean / (1.0 - s_mean)
    )
    
    U_post = fem_solver_batched(
        raw_E_from_Emean.unsqueeze(0)
    )[0]

# ============================================================
# 5. RESHAPE E FIELDS
# ============================================================

E_true_grid = (
    E_true.cpu()
    .numpy()
    .reshape(NY, NX)
)

E_mean_grid = (
    E_mean.cpu()
    .numpy()
    .reshape(NY, NX)
)

E_err_grid = np.abs(
    E_mean_grid - E_true_grid
)

# ============================================================
# 6. RESHAPE DISPLACEMENTS
# ============================================================
U_true_flat = U_true[0]
Ux_true = (
    U_true_flat[0::2]
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Uy_true = (
    U_true_flat[1::2]
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Ux_post = (
    U_post[0::2]
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Uy_post = (
    U_post[1::2]
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Ux_err = np.abs(
    Ux_post - Ux_true
)

Uy_err = np.abs(
    Uy_post - Uy_true
)

# ============================================================
# 7. E FIELD PLOTS (CONTOUR PLOTS)
# ============================================================

fig, ax = plt.subplots(
    1,
    3,
    figsize=(18,5)
)

emin = min(
    E_true_grid.min(),
    E_mean_grid.min()
)

emax = max(
    E_true_grid.max(),
    E_mean_grid.max()
)

# Define shared levels for consistent color scaling between True and Mean
levels_E = np.linspace(emin/1e9, emax/1e9, 50)

im = ax[0].contourf(
    E_true_grid/1e9,
    levels=levels_E,
    cmap="viridis"
)
ax[0].set_title("True E (GPa)")
fig.colorbar(im, ax=ax[0])

im = ax[1].contourf(
    E_mean_grid/1e9,
    levels=levels_E,
    cmap="viridis"
)
ax[1].set_title("Posterior Mean E (GPa)")
fig.colorbar(im, ax=ax[1])

# For error plots, let matplotlib determine levels automatically or use a fixed number
im = ax[2].contourf(
    E_err_grid/1e9,
    levels=50,
    cmap="plasma"
)
ax[2].set_title("|E Error| (GPa)")
fig.colorbar(im, ax=ax[2])

plt.tight_layout()
plt.show()

# ============================================================
# 8. Ux / Uy COMPARISON (CONTOUR PLOTS)
# ============================================================

fig, ax = plt.subplots(
    2,
    3,
    figsize=(18,10)
)

# -----------------------------
# Ux
# -----------------------------

uxmin = min(
    Ux_true.min(),
    Ux_post.min()
)

uxmax = max(
    Ux_true.max(),
    Ux_post.max()
)

levels_ux = np.linspace(uxmin, uxmax, 50)

im = ax[0,0].contourf(
    Ux_true,
    levels=levels_ux,
    cmap="viridis"
)
ax[0,0].set_title("True Ux")
fig.colorbar(im, ax=ax[0,0])

im = ax[0,1].contourf(
    Ux_post,
    levels=levels_ux,
    cmap="viridis"
)
ax[0,1].set_title("Posterior Mean Ux")
fig.colorbar(im, ax=ax[0,1])

im = ax[0,2].contourf(
    Ux_err,
    levels=50,
    cmap="plasma"
)
ax[0,2].set_title("|Ux Error|")
fig.colorbar(im, ax=ax[0,2])

# -----------------------------
# Uy
# -----------------------------

uymin = min(
    Uy_true.min(),
    Uy_post.min()
)

uymax = max(
    Uy_true.max(),
    Uy_post.max()
)

levels_uy = np.linspace(uymin, uymax, 50)

im = ax[1,0].contourf(
    Uy_true,
    levels=levels_uy,
    cmap="viridis"
)
ax[1,0].set_title("True Uy")
fig.colorbar(im, ax=ax[1,0])

im = ax[1,1].contourf(
    Uy_post,
    levels=levels_uy,
    cmap="viridis"
)
ax[1,1].set_title("Posterior Mean Uy")
fig.colorbar(im, ax=ax[1,1])

im = ax[1,2].contourf(
    Uy_err,
    levels=50,
    cmap="plasma"
)
ax[1,2].set_title("|Uy Error|")
fig.colorbar(im, ax=ax[1,2])

plt.tight_layout()
plt.show()

# ============================================================
# 9. NUMERICAL ERROR METRICS
# ============================================================

E_rel = (
    np.linalg.norm(E_mean_grid - E_true_grid)
    /
    np.linalg.norm(E_true_grid)
)

Ux_rel = (
    np.linalg.norm(Ux_post - Ux_true)
    /
    np.linalg.norm(Ux_true)
)

Uy_rel = (
    np.linalg.norm(Uy_post - Uy_true)
    /
    np.linalg.norm(Uy_true)
)

print("\nRelative Errors")
print("----------------")
print("E  :", E_rel)
print("Ux :", Ux_rel)
print("Uy :", Uy_rel)

print("\nMaximum Absolute Errors")
print("-----------------------")
print("E  :", E_err_grid.max()/1e9, "GPa")
print("Ux :", Ux_err.max())
print("Uy :", Uy_err.max())


# In[8]:


post_std = (
    E_samples.std(0)
    .cpu()
    .numpy()
    / 1e9
)

print(post_std.mean())
print(post_std.min())
print(post_std.max())


# In[9]:


post_mean = E_samples.mean(0)
post_std  = E_samples.std(0)

z = torch.abs(
    E_true - post_mean
) / (post_std + 1e-8)

print("Mean z =", z.mean().item())
print("Max z =", z.max().item())

print(
    "Inside 1σ =",
    (z < 1).float().mean().item()
)

print(
    "Inside 2σ =",
    (z < 2).float().mean().item()
)

print(
    "Inside 3σ =",
    (z < 3).float().mean().item()
)


# In[10]:


slope, intercept = np.polyfit(
    E_true.detach().cpu().numpy(),
    post_mean.detach().cpu().numpy(),
    1
)

corr = np.corrcoef(
    E_true.detach().cpu().numpy(),
    post_mean.detach().cpu().numpy()
)[0,1]
print(slope, intercept, corr)


# In[11]:


import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# POSTERIOR SAMPLES
# =====================================================

E_post = (
    E_samples
    .detach()
    .cpu()
    .numpy()
    / 1e9
)

# =====================================================
# TRUE FIELD
# =====================================================

E_true_plot = (
    E_true
    .detach()
    .cpu()
    .numpy()
    / 1e9
)

N_nodes = E_post.shape[1]

# =====================================================
# HORIZONTAL POSTERIOR BOXPLOTS
# =====================================================

fig, ax = plt.subplots(
    figsize=(12,60)
)

bp = ax.boxplot(
    [E_post[:, i] for i in range(N_nodes)],
    positions=np.arange(N_nodes),
    widths=0.6,
    vert=False,
    patch_artist=True,
    showfliers=False
)

for patch in bp['boxes']:
    patch.set_alpha(0.6)

# =====================================================
# TRUE VALUES
# =====================================================

ax.scatter(
    E_true_plot,
    np.arange(N_nodes),
    color='red',
    s=12,
    label='Truth',
    zorder=10
)

# =====================================================
# POSTERIOR MEAN
# =====================================================

E_post_mean = E_post.mean(axis=0)

ax.scatter(
    E_post_mean,
    np.arange(N_nodes),
    color='black',
    s=8,
    label='Posterior Mean',
    zorder=10
)

# =====================================================
# LABELS
# =====================================================

ax.set_xlabel("Young's Modulus (GPa)")
ax.set_ylabel("Node ID")

ax.set_title(
    "Posterior Distribution at Each Node"
)

ax.legend()

plt.tight_layout()
plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CHOOSE 6 NODES
# =====================================================

# nodes_to_plot = [
#     (0,0),
#     (0,NX-1),
#     (NY//2,NX//2),
#     (NY-1,0),
#     (NY-1,NX-1),
#     (3*NY//4,3*NX//4),
# ]
nodes_to_plot = torch.randint(0,16,(10,2))

# =====================================================
# PRIOR SAMPLES
# =====================================================

NPRIOR = E_samples.shape[0]

with torch.no_grad():

    raw_prior = sample_theta(NPRIOR)

    E_prior_all = (
        E_MIN
        + (E_MAX - E_MIN)
        * torch.sigmoid(raw_prior)
    )

    E_prior_all = (
        E_prior_all.cpu().numpy()
        / 1e9
    )

# =====================================================
# PLOT
# =====================================================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(18,10)
)

axes = axes.flatten()

for ax, (iy, ix) in zip(axes, nodes_to_plot):

    node_id = iy * NX + ix

    # posterior samples
    E_post = (
        E_samples[:, node_id]
        .detach()
        .cpu()
        .numpy()
        / 1e9
    )

    # prior samples
    E_prior = (
        E_prior_all[:, node_id]
    )

    # truth
    E_true_node = (
        E_true[node_id]
        .item()
        / 1e9
    )

    # posterior mean
    E_post_mean = E_post.mean()

    # posterior std
    E_post_std = E_post.std()
    xmin = min(
        E_prior.min(),
        E_post.min()
    )
    
    xmax = max(
        E_prior.max(),
        E_post.max()
    )
    
    bins = np.linspace(xmin,xmax,40)

    ax.hist(
        E_prior,
        bins=bins,
        density=True,
        alpha=0.35,
        label="Prior"
    )

    ax.hist(
        E_post,
        bins=bins,
        density=True,
        alpha=0.35,
        label="Posterior"
    )

    ax.axvline(
        E_true_node,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Truth"
    )

    ax.axvline(
        E_post_mean,
        color="black",
        linewidth=2,
        label="Post Mean"
    )

    ax.set_title(
        f"Node ({ix},{iy})\n"
        f"μ={E_post_mean:.1f}, σ={E_post_std:.1f} GPa"
    )

    ax.set_xlabel("E (GPa)")
    ax.set_ylabel("Density")

# single legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout()
plt.show()


# In[20]:


E_true_np = E_true.cpu().numpy() / 1e9
E_mean_np = E_mean.cpu().numpy() / 1e9

plt.figure(figsize=(6,6))

plt.scatter(
    E_true_np,
    E_mean_np,
    alpha=0.6
)

m = min(E_true_np.min(), E_mean_np.min())
M = max(E_true_np.max(), E_mean_np.max())

plt.plot([m,M],[m,M],'r--')

plt.xlabel("True E (GPa)")
plt.ylabel("Posterior Mean E (GPa)")
plt.axis("equal")
plt.grid(True)
plt.show()


# # bayesflow comparison

# In[2]:


model_mle = ConditionalINN().to(device)

optimizer_mle = torch.optim.AdamW(
    model_mle.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


# In[3]:


losses = []
for epoch in range(10000):

    idx = torch.randint(
        0,
        len(theta_dataset),
        (BATCH_SIZE,)
    )

    theta = theta_dataset[idx].to(device)

    y = normalize_y(
        y_dataset[idx].to(device)
    )

    z, log_det = model_mle(theta, y)

    log_pz = -0.5 * torch.sum(
        z**2,
        dim=1
    )

    logq = log_pz + log_det
    
    loss = -logq.mean()
    losses.append(loss.item())
    if loss <-200.0:
        break
    
    optimizer_mle.zero_grad()
    loss.backward()
    optimizer_mle.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss={loss.item():.4f}")


# In[5]:


import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()


# In[8]:


# # ============================================================
# # 6. EVALUATION ON A COMPLETELY NEW OBSERVATION
# # ============================================================

# import matplotlib.pyplot as plt

# # model.eval()

# N_POST = 2000
# N_PRIOR = 5000
# POST_BATCH_SIZE = 128
# N_RANDOM_NODES = 8

# # ------------------------------------------------------------
# # 1. Generate a NEW independent test material field
# # ------------------------------------------------------------

# with torch.no_grad():

#     # Fresh draw from the Gaussian prior
#     raw_E_test = sample_theta(1)

#     # Corresponding physical Young's modulus
#     E_true = (
#         E_MIN
#         + (E_MAX - E_MIN) * torch.sigmoid(raw_E_test)
#     )

#     # Generate displacement using the true material field
#     U_test = fem_solver_batched(raw_E_test)

#     # Generate strain measurements
#     y_test_clean = measure_batched(U_test)

#     # Add measurement noise, consistent with training data
#     y_test = y_test_clean + 1e-6 * torch.randn_like(y_test_clean)

#     # Normalize using TRAINING dataset statistics
#     y_test_normalized = normalize_y(y_test)

# print("New test observation shape:", y_test.shape)
# print("New true material field shape:", E_true.shape)


# In[9]:


import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# BAYESFLOW POSTERIOR SAMPLING
# ==========================================

model_mle.eval()

N_POST = 2000
POST_BATCH_SIZE = 128

bayesflow_samples = []

with torch.no_grad():

    y_condition = normalize_y(y_test)

    for start in range(0, N_POST, POST_BATCH_SIZE):

        current_batch = min(
            POST_BATCH_SIZE,
            N_POST - start
        )

        # Gaussian latent samples
        z = torch.randn(
            current_batch,
            THETA_DIM,
            device=device
        )

        # Same unseen observation for every posterior sample
        y_batch = y_condition.repeat(
            current_batch, 1
        )

        # Inverse conditional flow
        raw_E_samples, _ = model_mle.sample(
            z,
            y_batch
        )

        # Transform to physical Young's modulus
        E_samples = (
            E_MIN
            + (E_MAX - E_MIN)
            * torch.sigmoid(raw_E_samples)
        )

        bayesflow_samples.append(
            E_samples.detach().cpu()
        )

bayesflow_samples = torch.cat(
    bayesflow_samples,
    dim=0
)

print("BayesFlow posterior samples:",
      bayesflow_samples.shape)

# ==========================================
# POSTERIOR STATISTICS
# ==========================================

E_bf_mean = bayesflow_samples.mean(dim=0)

E_bf_std = bayesflow_samples.std(
    dim=0,
    unbiased=True
)

print("Posterior mean shape:", E_bf_mean.shape)
print("Posterior std shape:", E_bf_std.shape)


# In[10]:


# ==========================================
# PREPARE TRUE AND PREDICTED FIELDS
# ==========================================

# Always reconstruct the true field from the original
# raw_E_test to avoid accidental shape inconsistencies.

E_true_vec = (
    E_MIN
    + (E_MAX - E_MIN)
    * torch.sigmoid(raw_E_test)
).detach().cpu().reshape(-1)

E_bf_mean = E_bf_mean.detach().cpu().reshape(-1)
E_bf_std = E_bf_std.detach().cpu().reshape(-1)

assert E_true_vec.numel() == N_NODES
assert E_bf_mean.numel() == N_NODES

# Convert Pa to GPa
E_true_GPa = E_true_vec.numpy() / 1e9
E_bf_mean_GPa = E_bf_mean.numpy() / 1e9
E_bf_std_GPa = E_bf_std.numpy() / 1e9

# Reshape into spatial fields
E_true_grid = E_true_GPa.reshape(NY, NX)
E_bf_mean_grid = E_bf_mean_GPa.reshape(NY, NX)
E_bf_std_grid = E_bf_std_GPa.reshape(NY, NX)

E_bf_error_grid = np.abs(
    E_bf_mean_grid - E_true_grid
)

print("True E range:",
      E_true_GPa.min(), E_true_GPa.max())

print("BayesFlow posterior mean range:",
      E_bf_mean_GPa.min(), E_bf_mean_GPa.max())


# In[11]:


# ==========================================
# MATERIAL FIELD COMPARISON
# ==========================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 5)
)

im0 = axes[0].imshow(
    E_true_grid,
    extent=[0, LX, 0, LY],
    origin="lower",
    aspect="equal",
    cmap="viridis",
    vmin=100,
    vmax=300
)

axes[0].set_title("True Young's Modulus")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

im1 = axes[1].imshow(
    E_bf_mean_grid,
    extent=[0, LX, 0, LY],
    origin="lower",
    aspect="equal",
    cmap="viridis",
    vmin=100,
    vmax=300
)

axes[1].set_title("BayesFlow Posterior Mean")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

im2 = axes[2].imshow(
    E_bf_error_grid,
    extent=[0, LX, 0, LY],
    origin="lower",
    aspect="equal",
    cmap="magma"
)

axes[2].set_title("Absolute Error")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")

fig.colorbar(
    im0, ax=axes[:2],
    label="Young's modulus (GPa)",
    shrink=0.85
)

fig.colorbar(
    im2, ax=axes[2],
    label="Absolute error (GPa)",
    shrink=0.85
)

plt.tight_layout()
plt.show()


# In[12]:


# ==========================================
# DISPLACEMENT COMPARISON
# ==========================================

# Convert posterior mean physical E back to raw coordinates
s_mean = (
    E_bf_mean.to(device) - E_MIN
) / (E_MAX - E_MIN)

eps = 1e-6

s_mean = torch.clamp(
    s_mean,
    eps,
    1.0 - eps
)

raw_E_bf_mean = (
    torch.log(s_mean)
    - torch.log1p(-s_mean)
).unsqueeze(0)

# True FEM displacement
with torch.no_grad():

    U_true = fem_solver_batched(
        raw_E_test.to(device)
    )

    U_bf_mean = fem_solver_batched(
        raw_E_bf_mean
    )

# Separate displacement components
ux_true = U_true[0, 0::2].cpu().numpy().reshape(NY, NX)
uy_true = U_true[0, 1::2].cpu().numpy().reshape(NY, NX)

ux_bf = U_bf_mean[0, 0::2].cpu().numpy().reshape(NY, NX)
uy_bf = U_bf_mean[0, 1::2].cpu().numpy().reshape(NY, NX)

# ==========================================
# PLOT
# ==========================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 9)
)

fields = [
    ux_true,
    ux_bf,
    np.abs(ux_true - ux_bf),
    uy_true,
    uy_bf,
    np.abs(uy_true - uy_bf)
]

titles = [
    r"True $u_x$",
    r"BayesFlow $u_x$",
    r"$|u_x^{true}-u_x^{BF}|$",
    r"True $u_y$",
    r"BayesFlow $u_y$",
    r"$|u_y^{true}-u_y^{BF}|$"
]

for ax, field, title in zip(
    axes.flatten(),
    fields,
    titles
):

    im = ax.imshow(
        field,
        extent=[0, LX, 0, LY],
        origin="lower",
        aspect="equal",
        cmap="viridis"
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()


# In[13]:


# ==========================================
# NODE-WISE SCATTER PLOT
# ==========================================

plt.figure(figsize=(6, 6))

plt.scatter(
    E_true_GPa,
    E_bf_mean_GPa,
    alpha=0.7,
    s=25
)

plt.plot(
    [100, 300],
    [100, 300],
    "r--",
    linewidth=2,
    label="Perfect prediction"
)

plt.xlabel("True Young's modulus (GPa)")
plt.ylabel("BayesFlow posterior mean (GPa)")

plt.title("BayesFlow: True vs Posterior Mean")

plt.xlim(100, 300)
plt.ylim(100, 300)

plt.legend()
plt.grid(alpha=0.3)
plt.axis("equal")

plt.tight_layout()
plt.show()


# In[17]:


# ==========================================
# PRIOR SAMPLES
# ==========================================

with torch.no_grad():

    raw_E_prior = sample_theta(N_POST)

    E_prior_samples = (
        E_MIN
        + (E_MAX - E_MIN)
        * torch.sigmoid(raw_E_prior)
    ).cpu()

# ==========================================
# SELECT NODES
# ==========================================

torch.manual_seed(42)

selected_nodes = torch.randperm(
    N_NODES
)[:8].tolist()

fig, axes = plt.subplots(
    2, 4,
    figsize=(17, 8)
)

axes = axes.flatten()

for ax, node_id in zip(axes, selected_nodes):

    prior_node = (
        E_prior_samples[:, node_id].numpy() / 1e9
    )

    posterior_node = (
        bayesflow_samples[:, node_id].numpy() / 1e9
    )

    true_value = E_true_GPa[node_id]
    posterior_mean = posterior_node.mean()

    ax.hist(
        prior_node,
        bins=35,
        density=True,
        alpha=0.4,
        label="Prior"
    )

    ax.hist(
        posterior_node,
        bins=35,
        density=True,
        alpha=0.6,
        label="BayesFlow posterior"
    )

    # Ground truth
    ax.axvline(
        true_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label="True value"
    )

    # Posterior mean
    ax.axvline(
        posterior_mean,
        color="black",
        linestyle="-",
        linewidth=2,
        label="Posterior mean"
    )

    ax.set_title(f"Node ({node_id % 16},{node_id // 16})")
    ax.set_xlabel("Young's modulus (GPa)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)

# ==========================================
# COMMON LEGEND
# ==========================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    fontsize=11,
    frameon=True,
    bbox_to_anchor=(0.5, 1.02)
)

# plt.suptitle(
#     "BayesFlow: Prior vs Posterior at Selected Nodes",
#     fontsize=15,
#     y=1.10
# )

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()


# In[77]:


# ==========================================
# POOLED DISTRIBUTIONS
# ==========================================

prior_pooled = (
    E_prior_samples.numpy().reshape(-1) / 1e9
)

posterior_pooled = (
    bayesflow_samples.numpy().reshape(-1) / 1e9
)

true_pooled = E_true_GPa

plt.figure(figsize=(8, 5))

plt.hist(
    prior_pooled,
    bins=60,
    density=True,
    alpha=0.4,
    label="Prior"
)

plt.hist(
    posterior_pooled,
    bins=60,
    density=True,
    alpha=0.6,
    label="BayesFlow posterior"
)

plt.hist(
    true_pooled,
    bins=30,
    density=True,
    histtype="step",
    linewidth=2,
    label="True material field"
)

plt.xlabel("Young's modulus (GPa)")
plt.ylabel("Density")

plt.title("Pooled Prior and BayesFlow Posterior")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[70]:


# ============================================================
# BAYESFLOW POSTERIOR SAMPLES FOR THE SAME TEST OBSERVATION
# ============================================================

model_mle.eval()

N_POST = 2000
POST_BATCH_SIZE = 128

bayesflow_samples = []

with torch.no_grad():

    y_condition = normalize_y(y_test)

    for start in range(0, N_POST, POST_BATCH_SIZE):

        current_batch = min(
            POST_BATCH_SIZE,
            N_POST - start
        )

        # Sample latent Gaussian variables
        z = torch.randn(
            current_batch,
            THETA_DIM,
            device=device
        )

        y_batch = y_condition.repeat(
            current_batch, 1
        )

        # Inverse flow: latent z -> material field
        raw_E_samples, _ = model_mle.sample(
            z,
            y_batch
        )

        # Transform into physical Young's modulus
        E_samples = (
            E_MIN
            + (E_MAX - E_MIN)
            * torch.sigmoid(raw_E_samples)
        )

        bayesflow_samples.append(E_samples.cpu())


bayesflow_samples = torch.cat(
    bayesflow_samples,
    dim=0
)

# Posterior statistics
E_bf_mean = bayesflow_samples.mean(dim=0)
E_bf_std = bayesflow_samples.std(dim=0)

# Convert to GPa
E_bf_mean_GPa = E_bf_mean.numpy() / 1e9
E_bf_std_GPa = E_bf_std.numpy() / 1e9

E_true_GPa = E_true.detach().cpu().reshape(-1).numpy() / 1e9

print("BayesFlow posterior samples:", bayesflow_samples.shape)

# ============================================================
# PLOT 1: BAYESFLOW POSTERIOR MEAN AND ABSOLUTE ERROR
# ============================================================

E_bf_error = torch.abs(
    E_bf_mean - E_true.cpu()
)

E_bf_error_grid = (
    E_bf_error.numpy().reshape(NY, NX) / 1e9
)

E_true_grid = E_true_GPa.reshape(NY, NX)
E_bf_mean_grid = E_bf_mean_GPa.reshape(NY, NX)

# Spatial L2 error
weights = torch.ones(NY, NX)

weights[0, :] *= 0.5
weights[-1, :] *= 0.5
weights[:, 0] *= 0.5
weights[:, -1] *= 0.5

error_grid = (
    E_bf_mean - E_true.cpu()
).reshape(NY, NX)

true_grid = E_true.cpu().reshape(NY, NX)

L2_bf = torch.sqrt(
    torch.sum(weights * error_grid**2) * dx * dy
)

L2_true = torch.sqrt(
    torch.sum(weights * true_grid**2) * dx * dy
)

relative_L2_bf = L2_bf / (L2_true + 1e-12)

print("BayesFlow L2 error:",
      L2_bf.item() / 1e9, "GPa")

print("BayesFlow relative L2 error:",
      relative_L2_bf.item())


# imshow plots
extent = [0, LX, 0, LY]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

im0 = axes[0].imshow(
    E_true_grid,
    extent=extent,
    origin="lower",
    cmap="viridis",
    aspect="equal",
    vmin=100,
    vmax=300
)

axes[0].set_title("True Young's modulus")
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$y$")
fig.colorbar(im0, ax=axes[0], label="$E$ (GPa)")


im1 = axes[1].imshow(
    E_bf_mean_grid,
    extent=extent,
    origin="lower",
    cmap="viridis",
    aspect="equal",
    vmin=100,
    vmax=300
)

axes[1].set_title("BayesFlow posterior mean")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$y$")
fig.colorbar(im1, ax=axes[1], label="$E$ (GPa)")


im2 = axes[2].imshow(
    E_bf_error_grid,
    extent=extent,
    origin="lower",
    cmap="inferno",
    aspect="equal"
)

axes[2].set_title(
    f"Absolute error\nRelative $L_2$ = {relative_L2_bf.item():.4f}"
)

axes[2].set_xlabel("$x$")
axes[2].set_ylabel("$y$")
fig.colorbar(im2, ax=axes[2], label="Error (GPa)")

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 2: DISPLACEMENTS AT TRUE E AND BAYESFLOW POSTERIOR MEAN
# ============================================================

# Convert BayesFlow posterior mean E to raw representation
s_bf = (
    E_bf_mean.to(device) - E_MIN
) / (E_MAX - E_MIN)

s_bf = torch.clamp(s_bf, 1e-6, 1.0 - 1e-6)

raw_E_bf_mean = (
    torch.log(s_bf) - torch.log1p(-s_bf)
).unsqueeze(0)

with torch.no_grad():

    U_bf_mean = fem_solver_batched(raw_E_bf_mean)

    # True displacement field from the same test observation
    U_true_test = U_test


ux_true = U_true_test[0, 0::2].reshape(NY, NX).cpu().numpy() * 1e3
uy_true = U_true_test[0, 1::2].reshape(NY, NX).cpu().numpy() * 1e3

ux_bf = U_bf_mean[0, 0::2].reshape(NY, NX).cpu().numpy() * 1e3
uy_bf = U_bf_mean[0, 1::2].reshape(NY, NX).cpu().numpy() * 1e3

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

fields = [
    (ux_true, "$u_x$ at true $E$"),
    (ux_bf, "$u_x$ at BayesFlow posterior mean"),
    (uy_true, "$u_y$ at true $E$"),
    (uy_bf, "$u_y$ at BayesFlow posterior mean")
]

for ax, (field, title) in zip(axes.flatten(), fields):

    im = ax.imshow(
        field,
        extent=extent,
        origin="lower",
        cmap="coolwarm",
        aspect="equal"
    )

    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    fig.colorbar(im, ax=ax, label="Displacement (mm)")

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 3: BAYESFLOW POSTERIOR CALIBRATION SCATTER
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    E_true_GPa,
    E_bf_mean_GPa,
    s=25,
    alpha=0.65,
    edgecolors="none",
    label="BayesFlow"
)

lower = min(
    E_true_GPa.min(),
    E_bf_mean_GPa.min()
)

upper = max(
    E_true_GPa.max(),
    E_bf_mean_GPa.max()
)

ax.plot(
    [lower, upper],
    [lower, upper],
    "r--",
    linewidth=2,
    label="Perfect agreement"
)

ax.set_xlabel("True $E$ (GPa)")
ax.set_ylabel("BayesFlow posterior mean $E$ (GPa)")

ax.set_title("BayesFlow: posterior mean versus true field")

ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 4: PRIOR AND BAYESFLOW POSTERIOR AT 8 RANDOM NODES
# ============================================================

# Use the same prior samples generated for the JKO comparison.
# If E_prior_GPa is not available, generate it as follows:

with torch.no_grad():

    raw_E_prior = sample_theta(5000)

    E_prior = (
        E_MIN
        + (E_MAX - E_MIN) * torch.sigmoid(raw_E_prior)
    )

E_prior_GPa = E_prior.cpu().numpy() / 1e9

bayesflow_GPa = bayesflow_samples.numpy() / 1e9

torch.manual_seed(123)

random_nodes = torch.randperm(N_NODES)[:8].tolist()

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for ax, node_id in zip(axes, random_nodes):

    ax.hist(
        E_prior_GPa[:, node_id],
        bins=35,
        density=True,
        alpha=0.45,
        color="steelblue",
        label="Prior"
    )

    ax.hist(
        bayesflow_GPa[:, node_id],
        bins=35,
        density=True,
        alpha=0.55,
        color="orange",
        label="BayesFlow posterior"
    )

    ax.axvline(
        E_true_GPa[node_id],
        color="red",
        linestyle="--",
        linewidth=2,
        label="True $E$"
    )

    x_coord = nodes[node_id, 0].item()
    y_coord = nodes[node_id, 1].item()

    ax.set_title(
        f"Node {node_id}\n"
        f"$(x,y)=({x_coord:.2f},{y_coord:.2f})$"
    )

    ax.set_xlabel("$E$ (GPa)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.02)
)

fig.suptitle(
    "BayesFlow: prior and posterior distributions at selected nodes",
    y=1.07
)

plt.tight_layout()
plt.show()


# ============================================================
# PLOT 5: PRIOR AND BAYESFLOW POSTERIOR OVER ALL NODES
# ============================================================

prior_all = E_prior_GPa.flatten()

posterior_all = bayesflow_GPa.flatten()

true_all = E_true_GPa.flatten()

bins = np.linspace(
    E_MIN / 1e9,
    E_MAX / 1e9,
    60
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    prior_all,
    bins=bins,
    density=True,
    alpha=0.4,
    color="steelblue",
    label="Prior"
)

ax.hist(
    posterior_all,
    bins=bins,
    density=True,
    alpha=0.5,
    color="orange",
    label="BayesFlow posterior"
)

ax.hist(
    true_all,
    bins=bins,
    density=True,
    histtype="step",
    linestyle="--",
    linewidth=2.2,
    color="red",
    label="True field"
)

ax.set_xlabel("Young's modulus $E$ (GPa)")
ax.set_ylabel("Probability density")

ax.set_title(
    "BayesFlow: prior and posterior distributions over all nodes"
)

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# In[55]:


NSAMPLES = 5000

with torch.no_grad():

    y_obs_cond = normalize_y(y_obs)

    z = torch.randn(
        NSAMPLES,
        THETA_DIM,
        device=device
    )

    raw_E_samples, _ = model_mle.sample(
        z,
        y_obs_cond.repeat(NSAMPLES, 1)
    )


# In[56]:


with torch.no_grad():

    E_samples = (
        E_MIN
        + (E_MAX - E_MIN)
        * torch.sigmoid(raw_E_samples)
    )
    E_mean = E_samples.mean(0)
    s_mean = (E_mean - E_MIN)/(E_MAX-E_MIN)
    raw_E_from_Emean = torch.log(s_mean/(1-s_mean))


# In[57]:


with torch.no_grad():

    U_post = fem_solver_batched(
        raw_E_from_Emean.unsqueeze(0)
    )[0]


# In[58]:


E_true_phys = (
    E_MIN
    + (E_MAX - E_MIN)
    * torch.sigmoid(raw_E_true[0])
)

U_true_flat = U_true[0]

Ux_true = U_true_flat[0::2]
Uy_true = U_true_flat[1::2]

Ux_post = U_post[0::2]
Uy_post = U_post[1::2]

E_rel = (
    torch.norm(E_mean - E_true_phys)
    /
    torch.norm(E_true_phys)
)

Ux_rel = (
    torch.norm(Ux_post - Ux_true)
    /
    torch.norm(Ux_true)
)

Uy_rel = (
    torch.norm(Uy_post - Uy_true)
    /
    torch.norm(Uy_true)
)

print("\nRelative Errors")
print("----------------")
print("E  :", E_rel.item())
print("Ux :", Ux_rel.item())
print("Uy :", Uy_rel.item())

# =====================================================
# MAXIMUM ABSOLUTE ERRORS
# =====================================================

E_max_abs = torch.max(
    torch.abs(E_mean - E_true_phys)
)

Ux_max_abs = torch.max(
    torch.abs(Ux_post - Ux_true)
)

Uy_max_abs = torch.max(
    torch.abs(Uy_post - Uy_true)
)

print("\nMaximum Absolute Errors")
print("-----------------------")
print(
    f"E  : {E_max_abs.item()/1e9:.6f} GPa"
)
print(
    f"Ux : {Ux_max_abs.item():.8e}"
)
print(
    f"Uy : {Uy_max_abs.item():.8e}"
)


# In[59]:


print("\nAvg Absolute Errors")
print("-----------------------")
print("E  :", torch.abs(E_mean - E_true_phys).mean()/1e9, "GPa")


# In[60]:


import numpy as np

E_true_plot = (
    E_true_phys
    .cpu()
    .numpy()
    .reshape(NY, NX)
) / 1e9

E_post_plot = (
    E_mean
    .cpu()
    .numpy()
    .reshape(NY, NX)
) / 1e9

Ux_true_plot = (
    Ux_true
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Ux_post_plot = (
    Ux_post
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Uy_true_plot = (
    Uy_true
    .cpu()
    .numpy()
    .reshape(NY, NX)
)

Uy_post_plot = (
    Uy_post
    .cpu()
    .numpy()
    .reshape(NY, NX)
)


# In[61]:


emin = min(
    E_true_plot.min(),
    E_post_plot.min()
)

emax = max(
    E_true_plot.max(),
    E_post_plot.max()
)

fig, ax = plt.subplots(
    1, 3,
    figsize=(18,5)
)

im0 = ax[0].imshow(
    E_true_plot,
    origin="lower",
    vmin=emin,
    vmax=emax
)
ax[0].set_title("True E (GPa)")
plt.colorbar(im0, ax=ax[0])

im1 = ax[1].imshow(
    E_post_plot,
    origin="lower",
    vmin=emin,
    vmax=emax
)
ax[1].set_title("Posterior Mean E (GPa)")
plt.colorbar(im1, ax=ax[1])

im2 = ax[2].imshow(
    np.abs(E_true_plot - E_post_plot),
    origin="lower"
)
ax[2].set_title("|E Error| (GPa)")
plt.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.show()

uxmin = min(
    Ux_true_plot.min(),
    Ux_post_plot.min()
)

uxmax = max(
    Ux_true_plot.max(),
    Ux_post_plot.max()
)

fig, ax = plt.subplots(
    1, 3,
    figsize=(18,5)
)

im0 = ax[0].imshow(
    Ux_true_plot,
    origin="lower",
    vmin=uxmin,
    vmax=uxmax
)
ax[0].set_title("True Ux")
plt.colorbar(im0, ax=ax[0])

im1 = ax[1].imshow(
    Ux_post_plot,
    origin="lower",
    vmin=uxmin,
    vmax=uxmax
)
ax[1].set_title("Posterior Mean Ux")
plt.colorbar(im1, ax=ax[1])

im2 = ax[2].imshow(
    np.abs(Ux_true_plot - Ux_post_plot),
    origin="lower"
)
ax[2].set_title("|Ux Error|")
plt.colorbar(im2, ax=ax[2])
plt.tight_layout()
plt.show()

uymin = min(
    Uy_true_plot.min(),
    Uy_post_plot.min()
)

uymax = max(
    Uy_true_plot.max(),
    Uy_post_plot.max()
)

fig, ax = plt.subplots(
    1, 3,
    figsize=(18,5)
)

im0 = ax[0].imshow(
    Uy_true_plot,
    origin="lower",
    vmin=uymin,
    vmax=uymax
)
ax[0].set_title("True Uy")
plt.colorbar(im0, ax=ax[0])

im1 = ax[1].imshow(
    Uy_post_plot,
    origin="lower",
    vmin=uymin,
    vmax=uymax
)
ax[1].set_title("Posterior Mean Uy")
plt.colorbar(im1, ax=ax[1])

im2 = ax[2].imshow(
    np.abs(Uy_true_plot - Uy_post_plot),
    origin="lower"
)
ax[2].set_title("|Uy Error|")
plt.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.show()



# In[62]:


import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. E FIELD PLOTS (CONTOUR PLOTS)
# ============================================================

emin = min(
    E_true_plot.min(),
    E_post_plot.min()
)

emax = max(
    E_true_plot.max(),
    E_post_plot.max()
)

fig, ax = plt.subplots(
    1, 3,
    figsize=(18,5)
)

# Shared levels for consistent color scale across True and Post
levels_E = np.linspace(emin, emax, 50)

im0 = ax[0].contourf(
    E_true_plot,
    levels=levels_E,
    cmap="viridis"
)
ax[0].set_title("True E (GPa)")
plt.colorbar(im0, ax=ax[0])

im1 = ax[1].contourf(
    E_post_plot,
    levels=levels_E,
    cmap="viridis"
)
ax[1].set_title("Posterior Mean E (GPa)")
plt.colorbar(im1, ax=ax[1])

im2 = ax[2].contourf(
    np.abs(E_true_plot - E_post_plot),
    levels=50,
    cmap="plasma"
)
ax[2].set_title("|E Error| (GPa)")
plt.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.show()


# ============================================================
# 2. Ux COMPARISON (CONTOUR PLOTS)
# ============================================================

uxmin = min(
    Ux_true_plot.min(),
    Ux_post_plot.min()
)

uxmax = max(
    Ux_true_plot.max(),
    Ux_post_plot.max()
)

fig, ax = plt.subplots(
    1, 3,
    figsize=(18,5)
)

# Shared levels for Ux
levels_ux = np.linspace(uxmin, uxmax, 50)

im0 = ax[0].contourf(
    Ux_true_plot,
    levels=levels_ux,
    cmap="viridis"
)
ax[0].set_title("True Ux")
plt.colorbar(im0, ax=ax[0])

im1 = ax[1].contourf(
    Ux_post_plot,
    levels=levels_ux,
    cmap="viridis"
)
ax[1].set_title("Posterior Mean Ux")
plt.colorbar(im1, ax=ax[1])

im2 = ax[2].contourf(
    np.abs(Ux_true_plot - Ux_post_plot),
    levels=50,
    cmap="plasma"
)
ax[2].set_title("|Ux Error|")
plt.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.show()


# ============================================================
# 3. Uy COMPARISON (CONTOUR PLOTS)
# ============================================================

uymin = min(
    Uy_true_plot.min(),
    Uy_post_plot.min()
)

uymax = max(
    Uy_true_plot.max(),
    Uy_post_plot.max()
)

fig, ax = plt.subplots(
    1, 3,
    figsize=(18,5)
)

# Shared levels for Uy
levels_uy = np.linspace(uymin, uymax, 50)

im0 = ax[0].contourf(
    Uy_true_plot,
    levels=levels_uy,
    cmap="viridis"
)
ax[0].set_title("True Uy")
plt.colorbar(im0, ax=ax[0])

im1 = ax[1].contourf(
    Uy_post_plot,
    levels=levels_uy,
    cmap="viridis"
)
ax[1].set_title("Posterior Mean Uy")
plt.colorbar(im1, ax=ax[1])

im2 = ax[2].contourf(
    np.abs(Uy_true_plot - Uy_post_plot),
    levels=50,
    cmap="plasma"
)
ax[2].set_title("|Uy Error|")
plt.colorbar(im2, ax=ax[2])

plt.tight_layout()
plt.show()


# In[63]:


post_std = (
    E_samples.std(0)
    .cpu()
    .numpy()
    / 1e9
)

print(post_std.mean())
print(post_std.min())
print(post_std.max())


# In[64]:


E_true_flat = (
    E_true_phys
    .cpu()
    .numpy()
    / 1e9
)

E_post_flat = (
    E_mean
    .cpu()
    .numpy()
    / 1e9
)

plt.figure(figsize=(6,6))

plt.scatter(
    E_true_flat,
    E_post_flat,
    alpha=0.5
)

m = min(
    E_true_flat.min(),
    E_post_flat.min()
)

M = max(
    E_true_flat.max(),
    E_post_flat.max()
)

plt.plot(
    [m,M],
    [m,M],
    'r--'
)

plt.xlabel("True E (GPa)")
plt.ylabel("Posterior Mean E (GPa)")
plt.title("Calibration")

plt.grid(True)
plt.show()


# In[65]:


slope, intercept = np.polyfit(
    E_true_flat,
    E_post_flat,
    1
)

corr = np.corrcoef(
    E_true_flat,
    E_post_flat,
)[0,1]
print(slope, intercept, corr)


# In[66]:


import numpy as np
import matplotlib.pyplot as plt
import torch

# =====================================================
# 6 NODES
# =====================================================

# nodes_to_plot = [
#     (0,0),
#     (0,NX-1),
#     (NY//2,NX//2),
#     (NY-1,0),
#     (NY-1,NX-1),
#     (3*NY//4,3*NX//4),
# ]

# =====================================================
# PRIOR SAMPLES
# =====================================================

NPRIOR = raw_E_samples.shape[0]

with torch.no_grad():

    raw_prior = sample_theta(NPRIOR)

raw_prior = raw_prior.cpu().numpy()

# =====================================================
# POSTERIOR
# =====================================================

raw_post = (
    raw_E_samples
    .detach()
    .cpu()
    .numpy()
)

raw_true = (
    raw_E_true[0]
    .detach()
    .cpu()
    .numpy()
)

# =====================================================
# PLOT
# =====================================================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(18,10)
)

axes = axes.flatten()

for ax, (iy, ix) in zip(axes, nodes_to_plot):

    node_id = iy * NX + ix

    # posterior samples at node (GPa)

    post_node = (
        E_MIN
        + (E_MAX - E_MIN)
        * (1.0 / (1.0 + np.exp(-raw_post[:, node_id])))
    ) / 1e9
    
    # prior samples at node (GPa)
    
    prior_node = (
        E_MIN
        + (E_MAX - E_MIN)
        * (1.0 / (1.0 + np.exp(-raw_prior[:, node_id])))
    ) / 1e9
    
    # true value at node (GPa)
    
    true_node = (
        E_MIN
        + (E_MAX - E_MIN)
        * (1.0 / (1.0 + np.exp(-raw_true[node_id])))
    ) / 1e9

    # -----------------------------
    # posterior mean
    # -----------------------------

    post_mean = post_node.mean()
    post_std = post_node.std()

    # -----------------------------
    # histograms
    # -----------------------------

    ax.hist(
        prior_node,
        bins=40,
        density=True,
        alpha=0.35,
        label="Prior"
    )

    ax.hist(
        post_node,
        bins=40,
        density=True,
        alpha=0.35,
        label="Posterior"
    )

    ax.axvline(
        true_node,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Truth"
    )

    ax.axvline(
        post_mean,
        color="black",
        linewidth=2,
        label="Posterior Mean"
    )

    ax.set_title(
        f"Node ({ix},{iy})\n"
        f"μ={post_mean:.1f}, σ={post_std:.1f} GPa"
    )
    
    ax.set_xlabel("Young's Modulus (GPa)")
    ax.set_ylabel("Density")

# =====================================================
# SINGLE LEGEND
# =====================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout()
plt.show()


# In[51]:


post_mean = E_samples.mean(0)
post_std  = E_samples.std(0)

z = torch.abs(
    E_true - post_mean
) / (post_std + 1e-8)

print("Mean z =", z.mean().item())
print("Max z =", z.max().item())

print(
    "Inside 1σ =",
    (z < 1).float().mean().item()
)

print(
    "Inside 2σ =",
    (z < 2).float().mean().item()
)

print(
    "Inside 3σ =",
    (z < 3).float().mean().item()
)


# In[ ]:




