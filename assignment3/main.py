import numpy as np
from os import path
import sys

# Add assignment2 scripts to path
path_assignment2 = path.abspath(path.join(__file__, '../../assignment2/scripts/'))
sys.path.insert(0, path_assignment2)

from gnss_ils_solver import ils_solver

path_data = path.abspath(path.join(__file__, '../data/'))
# print("Data path:", path_data)

# Spacecraft and environmental parameters
m = 600 # kg
A = 1.0 # m^2
Cd = 2.6 
omega_E = 7.292115e-05 # rad s^-1
C20 = -4.841692151273e-04
rho = 1.0e-11 # kg m^-3

# Constants
c = 299792458 # m s^-1
GM = 3.986004415e05 # km^3 s^-2
mu = GM*1e9 # m^3 s^-2
R_E = 6378.1366 # km

# %% Part 1: Kalman Filter Problem Definition
sigma_pos = 2.0 # m
sigma_vel = 0.1 # m s^-1
pseudo_range_obs = 3 # m
corr_coeff = 0.7

# covariance matrix of the initial state vector
P_xx = np.diag([sigma_pos**2, sigma_pos**2, sigma_pos**2, sigma_vel**2, sigma_vel**2, sigma_vel**2])
P_xx[0, 3] = corr_coeff * sigma_pos * sigma_vel
P_xx[3, 0] = P_xx[0, 3]
P_xx[1, 4] = corr_coeff * sigma_pos * sigma_vel
P_xx[4, 1] = P_xx[1, 4]
P_xx[2, 5] = corr_coeff * sigma_pos * sigma_vel
P_xx[5, 2] = P_xx[2, 5]

print("P_xx:\n", P_xx)

# covariance matrix of the observations
R_yy = np.diag([pseudo_range_obs**2, pseudo_range_obs**2, pseudo_range_obs**2, pseudo_range_obs**2])

print("R_yy:\n", R_yy)


# state transition matrix
t = np.loadtxt(path.join(path_data, 't.txt'))
delta_t = t[1] - t[0]
# x_gps = np.loadtxt(path.join(path_data, 'rx_gps.txt'))
# y_gps = np.loadtxt(path.join(path_data, 'ry_gps.txt'))
# z_gps = np.loadtxt(path.join(path_data, 'rz_gps.txt'))

rx = np.loadtxt(path.join(path_data, 'rx.txt')) # km
ry = np.loadtxt(path.join(path_data, 'ry.txt')) # km
rz = np.loadtxt(path.join(path_data, 'rz.txt')) # km
vx = np.loadtxt(path.join(path_data, 'vx.txt')) # km s^-1
vy = np.loadtxt(path.join(path_data, 'vy.txt')) # km s^-1
vz = np.loadtxt(path.join(path_data, 'vz.txt')) # km s^-1
r_vec = np.concatenate((rx[:, np.newaxis], ry[:, np.newaxis], rz[:, np.newaxis]), axis=1)
r_norm = np.sqrt(rx**2 + ry**2 + rz**2)
v = np.concatenate((vx[:, np.newaxis], vy[:, np.newaxis], vz[:, np.newaxis]), axis=1)
v_norm = np.sqrt(vx**2 + vy**2 + vz**2)
# prn = np.loadtxt(path.join(path_data, 'PRN_ID.txt'))
# ... (Previous code: imports, parameters, data loading remain exactly as you have them)

# Combine loaded data into the state array (Size: N x 6)
y = np.array([rx, ry, rz, vx, vy, vz]).T

# --- DEFINING THE INTEGRATOR FOR VARIATIONAL EQUATIONS ---

def variational_equations(Y, mu, omega_E):
    """
    Computes derivatives for both State (6x1) and STM (6x6).
    Y: vector of size 42 [rx, ry, rz, vx, vy, vz, phi_00, ... phi_55]
    """
    # 1. Unpack State and STM
    r = Y[0:3]       # Position (meters)
    v = Y[3:6]       # Velocity (m/s)
    Phi = Y[6:].reshape((6, 6)) # STM (6x6)
    
    r_norm = np.linalg.norm(r)
    
    # 2. Physics Derivatives (Equations of Motion)
    # Omega vector (Earth rotation around Z)
    w_vec = np.array([0, 0, omega_E])
    
    # Accelerations
    # Gravity: -mu/r^3 * r
    a_grav = -(mu / r_norm**3) * r
    # Coriolis: -2(w x v)
    acc_cor = -2 * np.cross(w_vec, v)
    # Centrifugal: -w x (w x r)
    acc_cen = -np.cross(w_vec, np.cross(w_vec, r))
    
    a_total = a_grav + acc_cor + acc_cen
    
    # State Derivative portion: [v, a]
    d_state = np.concatenate((v, a_total))
    
    # 3. STM Derivatives (Linearized Dynamics F)
    # We construct the Jacobian F at this specific instant
    F = np.zeros((6, 6))
    
    # -- Top Right: Identity --
    F[0:3, 3:6] = np.identity(3)
    
    # -- Bottom Right: Coriolis (-2*Omega) --
    Omega_skew = np.array([
        [0, -omega_E, 0],
        [omega_E, 0, 0],
        [0, 0, 0]
    ])
    F[3:6, 3:6] = -2 * Omega_skew
    
    # -- Bottom Left: Gravity Gradient + Centrifugal --
    # Gravity Gradient: (mu/r^5) * (3*r*r^T - r^2*I)
    I3 = np.identity(3)
    r_outer = np.outer(r, r)
    grav_grad = (mu / r_norm**5) * (3 * r_outer - (r_norm**2 * I3))
    
    # Centrifugal Gradient: -Omega^2 (which is -Omega_skew @ Omega_skew)
    centrifugal_grad = - (Omega_skew @ Omega_skew)
    
    F[3:6, 0:3] = grav_grad + centrifugal_grad
    
    # STM Derivative: dPhi = F * Phi
    d_Phi = F @ Phi
    
    # 4. Repack into size 42
    d_Y = np.concatenate((d_state, d_Phi.flatten()))
    return d_Y

def rk4_step(Y, dt, mu, omega_E):
    """ Standard Runge-Kutta 4 Integrator """
    k1 = variational_equations(Y, mu, omega_E)
    k2 = variational_equations(Y + 0.5 * dt * k1, mu, omega_E)
    k3 = variational_equations(Y + 0.5 * dt * k2, mu, omega_E)
    k4 = variational_equations(Y + dt * k3, mu, omega_E)
    
    return Y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# --- EXECUTION: INTEGRATE FROM EPOCH 0 TO EPOCH 1 ---

# 1. Prepare Initial State (Epoch 0)
# CRITICAL: Convert input data (km) to meters to match 'mu'
r0 = y[0, 0:3] * 1000.0 
v0 = y[0, 3:6] * 1000.0

# 2. Prepare Initial STM
Phi0 = np.identity(6)

# 3. Combine into Y vector (Size 42)
Y0 = np.concatenate((r0, v0, Phi0.flatten()))

# 4. Integrate
# delta_t was already calculated in your code as t[1] - t[0]
print(f"Integrating over time step: {delta_t} seconds...")
Y_next = rk4_step(Y0, delta_t, mu, omega_E)

# 5. Extract Results
r_next = Y_next[0:3] # meters
v_next = Y_next[3:6] # m/s
Phi_next = Y_next[6:].reshape((6, 6))

# --- REPORTING ---
# Convert to km and km/s for output
r_next_km = r_next / 1000.0  # meters to km
v_next_km = v_next / 1000.0  # m/s to km/s

print("\n" + "="*40)
print(f"INTEGRATED STATE VECTOR (Epoch t={t[1]})")
print("="*40)
print(f"Position X: {r_next_km[0]:.10f} km")
print(f"Position Y: {r_next_km[1]:.10f} km")
print(f"Position Z: {r_next_km[2]:.10f} km")
print("-" * 20)
print(f"Velocity X: {v_next_km[0]:.10f} km/s")
print(f"Velocity Y: {v_next_km[1]:.10f} km/s")
print(f"Velocity Z: {v_next_km[2]:.10f} km/s")

# print("\n" + "="*40)
# print("INTEGRATED STM (Sample Elements)")
# print("="*40)
# # Printing corner elements as a check
# print(f"Phi[0,0]: {Phi_next[0,0]:.10f}")
# print(f"Phi[0,5]: {Phi_next[0,5]:.10f}")
# print(f"Phi[5,0]: {Phi_next[5,0]:.10f}")
# print(f"Phi[5,5]: {Phi_next[5,5]:.10f}")

# Verification check against provided file data for Epoch 1
print("\n" + "="*40)
print("COMPARISON WITH DATA FILE (Epoch 1)")
print("="*40)
r_file_km = y[1, 0:3]  # already in km
diff_km = r_next_km - r_file_km
print(f"Difference (Integrated - File): {np.linalg.norm(diff_km):.10f} km")