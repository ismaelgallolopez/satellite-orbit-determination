"""
Least-Squares Orbit Determination Calculations
GRACE-FO LEO Satellite using GPS Pseudo-range Observations

Task 1: Compute Observation Covariance Matrix (P_yy)
Task 2: Linearise and Evaluate a Single Pseudo-range Observation
"""

import numpy as np

# Set print options for better formatting
np.set_printoptions(precision=6, suppress=True)

print("="*80)
print("LEAST-SQUARES ORBIT DETERMINATION CALCULATIONS")
print("="*80)

# ============================================================================
# TASK 1: OBSERVATION COVARIANCE MATRIX (P_yy)
# ============================================================================
print("\n" + "="*80)
print("TASK 1: OBSERVATION COVARIANCE MATRIX (P_yy)")
print("="*80)

# Given parameters
n_obs = 4  # Number of observations
sigma = 3.0  # Standard deviation [m]
rho = 0.2  # Correlation coefficient

print(f"\nGiven:")
print(f"  Number of observations (n):   {n_obs}")
print(f"  Standard deviation (σ):       {sigma} m")
print(f"  Correlation coefficient (ρ):  {rho}")

# Compute variance
variance = sigma**2  # [m^2]

# Construct the covariance matrix
# P_ii = σ^2 (variance on diagonal)
# P_ij = ρ * σ_i * σ_j (covariance off-diagonal)
P_yy = np.zeros((n_obs, n_obs))

for i in range(n_obs):
    for j in range(n_obs):
        if i == j:
            P_yy[i, j] = variance  # Diagonal: variance
        else:
            P_yy[i, j] = rho * sigma * sigma  # Off-diagonal: covariance

print(f"\nObservation Covariance Matrix P_yy [{n_obs}×{n_obs}] (m²):")
print(P_yy)

print(f"\n>>> ANSWER (Question 9):")
print(f"    Diagonal elements (P_ii):     {variance:.2f} m²")
print(f"    Off-diagonal elements (P_ij): {rho * sigma * sigma:.2f} m²")

# ============================================================================
# TASK 2: LINEARISE A SINGLE PSEUDO-RANGE OBSERVATION
# ============================================================================
print("\n" + "="*80)
print("TASK 2: LINEARISE A SINGLE PSEUDO-RANGE OBSERVATION")
print("="*80)

# Constants
c = 299792.458  # Speed of light [km/s]

# Initial receiver state (GRACE-FO LEO satellite)
x_0 = 6878.0  # [km]
y_0 = 0.0     # [km]
z_0 = 0.0     # [km]
dt_r_0 = 0.001  # Receiver clock offset [s]

# PRN (GPS satellite) position
x_PRN = -371.9349049   # [km]
y_PRN = 19871.754959   # [km]
z_PRN = 17630.753853   # [km]
dt_t = 5.977963643e-5  # Transmitter clock offset [s]

print(f"\nGiven Constants:")
print(f"  Speed of light (c):           {c} km/s")

print(f"\nInitial Receiver State (GRACE-FO):")
print(f"  x_0:                          {x_0} km")
print(f"  y_0:                          {y_0} km")
print(f"  z_0:                          {z_0} km")
print(f"  δt_r,0:                       {dt_r_0} s")

print(f"\nPRN (GPS Satellite) State:")
print(f"  x_PRN:                        {x_PRN} km")
print(f"  y_PRN:                        {y_PRN} km")
print(f"  z_PRN:                        {z_PRN} km")
print(f"  δt_t:                         {dt_t} s")

# ============================================================================
# CALCULATION 1: Geometric Range (R_0)
# ============================================================================
print(f"\n" + "-"*80)
print("CALCULATION 1: Geometric Range (R_0)")
print("-"*80)

# Position difference vector
dx = x_0 - x_PRN
dy = y_0 - y_PRN
dz = z_0 - z_PRN

print(f"\nPosition difference vector:")
print(f"  Δx = x_0 - x_PRN = {x_0} - ({x_PRN}) = {dx:.6f} km")
print(f"  Δy = y_0 - y_PRN = {y_0} - ({y_PRN}) = {dy:.6f} km")
print(f"  Δz = z_0 - z_PRN = {z_0} - ({z_PRN}) = {dz:.6f} km")

# Geometric range
R_0 = np.sqrt(dx**2 + dy**2 + dz**2)

print(f"\nGeometric Range:")
print(f"  R_0 = √((Δx)² + (Δy)² + (Δz)²)")
print(f"  R_0 = √({dx:.6f}² + {dy:.6f}² + {dz:.6f}²)")
print(f"  R_0 = √{dx**2 + dy**2 + dz**2:.6f}")
print(f"  R_0 = {R_0:.6f} km")

print(f"\n>>> ANSWER: R_0 = {R_0:.6f} km = {R_0*1000:.3f} m")

# ============================================================================
# CALCULATION 2: Calculated Pseudo-range f(x_0)
# ============================================================================
print(f"\n" + "-"*80)
print("CALCULATION 2: Calculated Pseudo-range f(x_0)")
print("-"*80)

# Clock offset contribution
clock_offset = c * (dt_r_0 - dt_t)

print(f"\nClock offset contribution:")
print(f"  c(δt_r - δt_t) = {c} × ({dt_r_0} - {dt_t})")
print(f"  c(δt_r - δt_t) = {c} × {dt_r_0 - dt_t}")
print(f"  c(δt_r - δt_t) = {clock_offset:.6f} km")

# Pseudo-range
f_x0 = R_0 + clock_offset

print(f"\nPseudo-range:")
print(f"  f(x_0) = R_0 + c(δt_r,0 - δt_t)")
print(f"  f(x_0) = {R_0:.6f} + {clock_offset:.6f}")
print(f"  f(x_0) = {f_x0:.6f} km")

print(f"\n>>> ANSWER: f(x_0) = {f_x0:.6f} km = {f_x0*1000:.3f} m")

# ============================================================================
# CALCULATION 3: Design Matrix H (1×4)
# ============================================================================
print(f"\n" + "-"*80)
print("CALCULATION 3: Design Matrix H (1×4)")
print("-"*80)

# Design matrix elements
H_11 = dx / R_0  # ∂ρ/∂x = (x_0 - x_PRN) / R_0
H_12 = dy / R_0  # ∂ρ/∂y = (y_0 - y_PRN) / R_0
H_13 = dz / R_0  # ∂ρ/∂z = (z_0 - z_PRN) / R_0
H_14 = c         # ∂ρ/∂(δt_r) = c

# Construct design matrix
H = np.array([[H_11, H_12, H_13, H_14]])

print(f"\nDesign matrix elements:")
print(f"  H_11 = (x_0 - x_PRN) / R_0 = {dx:.6f} / {R_0:.6f} = {H_11:.6f}")
print(f"  H_12 = (y_0 - y_PRN) / R_0 = {dy:.6f} / {R_0:.6f} = {H_12:.6f}")
print(f"  H_13 = (z_0 - z_PRN) / R_0 = {dz:.6f} / {R_0:.6f} = {H_13:.6f}")
print(f"  H_14 = c = {H_14:.6f} km/s")

print(f"\nDesign Matrix H [1×4]:")
print(H)

print(f"\n>>> ANSWER (Question 13):")
print(f"    H_11 (unitless):  {H_11:.6f}")
print(f"    H_12 (unitless):  {H_12:.6f}")
print(f"    H_13 (unitless):  {H_13:.6f}")
print(f"    H_14 (km/s):      {H_14:.6f}")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print(f"\nTASK 1: Observation Covariance Matrix P_yy [4×4] (m²):")
print(P_yy)
print(f"  Diagonal elements:     {variance:.2f} m²")
print(f"  Off-diagonal elements: {rho * sigma * sigma:.2f} m²")

print(f"\nTASK 2: Linearisation Results:")
print(f"  Geometric Range R_0:       {R_0:.6f} km")
print(f"  Pseudo-range f(x_0):       {f_x0:.6f} km")
print(f"  Design Matrix H [1×4]:")
print(f"    H_11 (unitless):         {H_11:.6f}")
print(f"    H_12 (unitless):         {H_12:.6f}")
print(f"    H_13 (unitless):         {H_13:.6f}")
print(f"    H_14 (km/s):             {H_14:.6f}")

print("="*80)
