import numpy as np
from os import path
import sys
import matplotlib.pyplot as plt

# --- 1. SETUP PATHS ---
path_data = path.abspath(path.join(__file__, '../data/'))

# --- 2. CONSTANTS & PARAMETERS ---
m = 600 # kg
omega_E = 7.292115e-05 # rad s^-1
# mu in km^3/s^2 (matching MATLAB)
mu = 3.986004415e05  # km^3/s^2 

# --- 3. KALMAN FILTER SETTINGS ---
sigma_pos = 2e-3 # km (2 meters)
sigma_vel = 0.1e-3 # km/s (0.1 m/s)
pseudo_range_obs_err = 3e-3 # km (3 meters)
corr_coeff = 0.7

# Initial Covariance P_xx
P_xx = np.diag([sigma_pos**2, sigma_pos**2, sigma_pos**2, sigma_vel**2, sigma_vel**2, sigma_vel**2])
P_xx[0, 3] = corr_coeff * sigma_pos * sigma_vel; P_xx[3, 0] = P_xx[0, 3]
P_xx[1, 4] = corr_coeff * sigma_pos * sigma_vel; P_xx[4, 1] = P_xx[1, 4]
P_xx[2, 5] = corr_coeff * sigma_pos * sigma_vel; P_xx[5, 2] = P_xx[2, 5]

print("\nCovariance matrix of the initial state vector (P_xx):")
print(P_xx) 

# --- 4. LOAD DATA ---
t = np.loadtxt(path.join(path_data, 't.txt'))
delta_t = t[1] - t[0]

# Precise Reference Orbit (for validation) - [km, km/s]
rx = np.loadtxt(path.join(path_data, 'rx.txt')) 
ry = np.loadtxt(path.join(path_data, 'ry.txt')) 
rz = np.loadtxt(path.join(path_data, 'rz.txt')) 
vx = np.loadtxt(path.join(path_data, 'vx.txt')) 
vy = np.loadtxt(path.join(path_data, 'vy.txt')) 
vz = np.loadtxt(path.join(path_data, 'vz.txt')) 

# Combine reference into array (N x 6) [km, km/s]
y_ref = np.array([rx, ry, rz, vx, vy, vz]).T

# GPS Data (for Measurement Update) - [km]
# Now we load these because we need them for the solver!
rx_gps = np.loadtxt(path.join(path_data, 'rx_gps.txt'))
ry_gps = np.loadtxt(path.join(path_data, 'ry_gps.txt'))
rz_gps = np.loadtxt(path.join(path_data, 'rz_gps.txt'))

# Pseudoranges - [km]
ca_range = np.loadtxt(path.join(path_data, 'CA_range.txt'))

# --- 5. OBSERVATION MODEL ---

def observation_model(x_receiver, x_satellites):
    """
    Observation model for pseudorange measurements (matching MATLAB)
    Inputs:
        x_receiver: 6x1 state [rx; ry; rz; vx; vy; vz] in km, km/s
        x_satellites: 6xN array of satellite states in km, km/s
    Outputs:
        h_x: N x 1 predicted pseudoranges in km
        H: N x 6 design matrix
    """
    N_sats = x_satellites.shape[1]
    h_x = np.zeros(N_sats)
    H = np.zeros((N_sats, 6))
    
    r_receiver = x_receiver[0:3]  # km
    
    for i in range(N_sats):
        r_sat = x_satellites[0:3, i]  # km
        delta_r = r_receiver - r_sat
        dist = np.linalg.norm(delta_r)
        h_x[i] = dist
        # Design matrix (gradient of h with respect to receiver position)
        H[i, 0:3] = delta_r / dist
    
    return h_x, H

# --- 6. DYNAMICS FUNCTIONS (2-BODY) ---

def variational_equations(Y, mu, omega_E):
    """ Derivatives for State (6x1) and STM (6x6) """
    r = Y[0:3] # km
    v = Y[3:6] # km/s
    Phi = Y[6:].reshape((6, 6))
    r_norm = np.linalg.norm(r)
    
    # Physics
    w_vec = np.array([0, 0, omega_E])
    a_grav = -(mu / r_norm**3) * r
    acc_cor = -2 * np.cross(w_vec, v)
    acc_cen = -np.cross(w_vec, np.cross(w_vec, r))
    a_total = a_grav + acc_cor + acc_cen
    
    # Jacobian F
    F = np.zeros((6, 6))
    F[0:3, 3:6] = np.identity(3)
    
    Omega_skew = np.array([[0, -omega_E, 0], [omega_E, 0, 0], [0, 0, 0]])
    F[3:6, 3:6] = -2 * Omega_skew
    
    I3 = np.identity(3)
    r_outer = np.outer(r, r)
    grav_grad = (mu / r_norm**5) * (3 * r_outer - (r_norm**2 * I3))
    centrifugal_grad = - (Omega_skew @ Omega_skew)
    
    F[3:6, 0:3] = grav_grad + centrifugal_grad
    d_Phi = F @ Phi
    
    return np.concatenate((np.concatenate((v, a_total)), d_Phi.flatten()))

def rk4_step(Y, dt, mu, omega_E):
    k1 = variational_equations(Y, mu, omega_E)
    k2 = variational_equations(Y + 0.5 * dt * k1, mu, omega_E)
    k3 = variational_equations(Y + 0.5 * dt * k2, mu, omega_E)
    k4 = variational_equations(Y + dt * k3, mu, omega_E)
    return Y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- 7. EKF IMPLEMENTATION ---

# Initialize with Reference at t0 (Epoch 1)
# State is in km, km/s (matching MATLAB)
x_est = y_ref[0].copy()
P_est = P_xx.copy()

# Storage
results_pos = []
results_vel = []
results_P = []

print(f"\nStarting EKF Loop over {len(t)} epochs...")

# Variable to store integrated state at second epoch
integrated_state_second_epoch = None
x_pred = x_est.copy()
P_pred = P_est.copy()

for k in range(len(t)):
    # ==========================
    # 1. MEASUREMENT UPDATE (MATLAB order: update first!)
    # ==========================
    try:
        # Get tracked satellites for this epoch
        tracked_mask = ca_range[k, :] > 0
        
        if np.sum(tracked_mask) >= 4:
            # Get pseudorange measurements (already in km)
            z_k = ca_range[k, tracked_mask]
            
            # Get GPS satellite states for tracked satellites (already in km, km/s)
            x_sats = np.array([
                rx_gps[k, tracked_mask],
                ry_gps[k, tracked_mask],
                rz_gps[k, tracked_mask],
                np.zeros(np.sum(tracked_mask)),  # velocities not used in observation
                np.zeros(np.sum(tracked_mask)),
                np.zeros(np.sum(tracked_mask))
            ])
            
            # Compute predicted measurements and design matrix
            h_x_k, H_k = observation_model(x_pred, x_sats)
            
            # Innovation (residual)
            dz_k = z_k - h_x_k
            
            # Measurement covariance
            R_k = (pseudo_range_obs_err**2) * np.eye(np.sum(tracked_mask))
            
            # Kalman Gain
            S = H_k @ P_pred @ H_k.T + R_k
            K_k = P_pred @ H_k.T @ np.linalg.inv(S)
            
            # Update State
            x_est = x_pred + K_k @ dz_k
            
            # Update Covariance
            P_est = (np.eye(6) - K_k @ H_k) @ P_pred
            
        else:
            # Not enough satellites
            x_est = x_pred.copy()
            P_est = P_pred.copy()

    except Exception as e:
        # Debugging: Print error if solver fails
        print(f"Update failed at step {k}: {e}")
        x_est = x_pred.copy()
        P_est = P_pred.copy()

    # Store results
    results_pos.append(x_est[0:3].copy())
    results_vel.append(x_est[3:6].copy())
    results_P.append(P_est.copy())
    
    # ==========================
    # 2. TIME UPDATE (PROPAGATION) - for next epoch
    # ==========================
    if k < len(t) - 1:  # Don't propagate after last epoch
        Phi_prev = np.identity(6)
        Y_input = np.concatenate((x_est, Phi_prev.flatten()))
        
        # Propagate to next step
        Y_pred = rk4_step(Y_input, delta_t, mu, omega_E)
        
        x_pred = Y_pred[0:6]
        Phi_k = Y_pred[6:].reshape((6, 6))
        
        # Propagate Covariance (Q=0)
        P_pred = Phi_k @ P_est @ Phi_k.T
        
        # Store integrated state at second epoch (k=0 means propagating from epoch 1 to epoch 2)
        if k == 0:
            integrated_state_second_epoch = x_pred.copy()

# Print integrated state vector at second epoch
if integrated_state_second_epoch is not None:
    print("\n" + "="*60)
    print("Integrated state vector at the second epoch (before measurement update):")
    print("="*60)
    print(f"Position (x, y, z) [km]: [{integrated_state_second_epoch[0]:.10f}, {integrated_state_second_epoch[1]:.10f}, {integrated_state_second_epoch[2]:.10f}]")
    print(f"Velocity (vx, vy, vz) [km/s]: [{integrated_state_second_epoch[3]:.10f}, {integrated_state_second_epoch[4]:.10f}, {integrated_state_second_epoch[5]:.10f}]")
    print("="*60)

# --- 7. REPORTING (Table) ---
indices_to_report = [9, 19, 29] # Epochs 10, 20, 30

print("\n" + "="*80)
print(f"{'Epoch':<10} | {'Position (x, y, z) [km]':<50} | {'Velocity (vx, vy, vz) [km/s]':<50}")
print("="*80)

for idx in indices_to_report:
    if idx < len(results_pos):
        pos = results_pos[idx]
        vel = results_vel[idx]
        
        # Format: km with 6 decimals (matching MATLAB)
        pos_str = f"[{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]"
        vel_str = f"[{vel[0]:.6f}, {vel[1]:.6f}, {vel[2]:.6f}]"
        print(f"{idx+1:<10} | {pos_str:<50} | {vel_str:<50}")
print("="*80)


# --- 8. PLOTTING (Tasks 7 & 8) ---
est_pos = np.array(results_pos) # (N, 3) in km
est_vel = np.array(results_vel) # (N, 3) in km/s
cov_history = np.array(results_P) # (N, 6, 6)

# Truth in km (already in km)
true_pos = y_ref[:, 0:3]
true_vel = y_ref[:, 3:6]

# Errors in km, convert to m for plotting
error_pos = np.linalg.norm(est_pos - true_pos, axis=1) * 1000.0  # m
error_vel = np.linalg.norm(est_vel - true_vel, axis=1) * 1000.0  # m/s

# Standard Deviations in km, convert to m for plotting
var_pos = cov_history[:, 0, 0] + cov_history[:, 1, 1] + cov_history[:, 2, 2]
sigma_pos_total = np.sqrt(var_pos) * 1000.0  # m
var_vel = cov_history[:, 3, 3] + cov_history[:, 4, 4] + cov_history[:, 5, 5]
sigma_vel_total = np.sqrt(var_vel) * 1000.0  # m/s

# Figure Task 7
fig7, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(t, error_pos, 'b-', label=r'$\delta r$')
ax1.set_ylabel('Position Error [m]')
ax1.set_title('Task 7: Errors (Estimated vs Precise)')
ax1.grid(True); ax1.legend()

ax2.plot(t, error_vel, 'r-', label=r'$\delta v$')
ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Velocity Error [m/s]')
ax2.grid(True); ax2.legend()
plt.tight_layout()
plt.savefig(path.join(path.dirname(__file__), 'plots/task7_errors.png'), dpi=300, bbox_inches='tight')
print('Saved: plots/task7_errors.png')

# Figure Task 8
fig8, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax3.plot(t, sigma_pos_total, 'b-', label=r'$\sigma_r$')
ax3.set_ylabel(r'Pos Std Dev [m]')
ax3.set_title('Task 8: Standard Deviations')
ax3.grid(True); ax3.legend()

ax4.plot(t, sigma_vel_total, 'r-', label=r'$\sigma_v$')
ax4.set_xlabel('Time [s]'); ax4.set_ylabel(r'Vel Std Dev [m/s]')
ax4.grid(True); ax4.legend()
plt.tight_layout()
plt.savefig(path.join(path.dirname(__file__), 'plots/task8_std_dev.png'), dpi=300, bbox_inches='tight')
print('Saved: plots/task8_std_dev.png')

# %% Part 3: Kalman Filter WITH Process Noise
print("\n" + "="*60)
print("STARTING PART 3: EKF WITH PROCESS NOISE")
print("="*60)

# --- 1. DEFINE PROCESS NOISE Q ---
# Based on MATLAB implementation
sigma_a = 0.01e-3  # km (Process noise for Position states)
sigma_b = 0.01e-3  # km/s (Process noise for Velocity states)

# Construct Q Matrix (Diagonal as per Appendix code [cite: 590])
# Q = diag([sigma_a^2, ..., sigma_b^2, ...])
q_diag = np.concatenate([
    np.full(3, sigma_a**2),
    np.full(3, sigma_b**2)
])
Q = np.diag(q_diag)

print(f"Process Noise Sigma A: {sigma_a}")
print(f"Process Noise Sigma B: {sigma_b}")
print("Q Matrix Diagonal:\n", q_diag)

# --- 2. RE-INITIALIZATION ---
x_est = y_ref[0].copy()  # Reset to initial truth (km, km/s)
P_est = P_xx.copy()      # Reset covariance

# Storage for new results
results_pos_Q = []
results_vel_Q = []
rms_residuals = [] # To store RMS for each epoch

# --- 3. EKF LOOP WITH Q ---
print(f"Running EKF with Process Noise over {len(t)} epochs...")

x_pred = x_est.copy()
P_pred = P_est.copy()

for k in range(len(t)):
    
    # === UPDATE ===
    current_rms = 0.0 # Default if no update
    
    try:
        # Get tracked satellites for this epoch
        tracked_mask = ca_range[k, :] > 0
        
        if np.sum(tracked_mask) >= 4:
            # Get pseudorange measurements (already in km)
            z_k = ca_range[k, tracked_mask]
            
            # Get GPS satellite states for tracked satellites (already in km, km/s)
            x_sats = np.array([
                rx_gps[k, tracked_mask],
                ry_gps[k, tracked_mask],
                rz_gps[k, tracked_mask],
                np.zeros(np.sum(tracked_mask)),
                np.zeros(np.sum(tracked_mask)),
                np.zeros(np.sum(tracked_mask))
            ])
            
            # Compute predicted measurements and design matrix
            h_x_k, H_k = observation_model(x_pred, x_sats)
            
            # Innovation (residual)
            dz_k = z_k - h_x_k
            
            # Measurement covariance
            R_k = (pseudo_range_obs_err**2) * np.eye(np.sum(tracked_mask))
            
            # Kalman Gain
            S = H_k @ P_pred @ H_k.T + R_k
            K_k = P_pred @ H_k.T @ np.linalg.inv(S)
            
            # Update State
            x_est = x_pred + K_k @ dz_k
            
            # Update Covariance
            P_est = (np.eye(6) - K_k @ H_k) @ P_pred
            
            # Calculate observation residual RMS (matching MATLAB)
            dx_k = K_k @ dz_k
            e_k = dz_k - H_k @ dx_k
            current_rms = np.sqrt(np.mean(e_k**2)) * 1000.0  # Convert km to m
            
        else:
            x_est = x_pred.copy()
            P_est = P_pred.copy()

    except Exception as e:
        print(f"Update failed at step {k}: {e}")
        x_est = x_pred.copy()
        P_est = P_pred.copy()

    # Store
    results_pos_Q.append(x_est[0:3].copy())
    results_vel_Q.append(x_est[3:6].copy())
    rms_residuals.append(current_rms)
    
    # === PREDICTION === (for next epoch)
    if k < len(t) - 1:
        Phi_prev = np.identity(6)
        Y_input = np.concatenate((x_est, Phi_prev.flatten()))
        Y_pred = rk4_step(Y_input, delta_t, mu, omega_E)
        
        x_pred = Y_pred[0:6]
        Phi_k = Y_pred[6:].reshape((6, 6))
        
        # Propagate Covariance WITH Q
        P_pred = Phi_k @ P_est @ Phi_k.T + Q

# --- 4. PROCESSING RESULTS ---
est_pos_Q = np.array(results_pos_Q)  # km
est_vel_Q = np.array(results_vel_Q)  # km/s
rms_history = np.array(rms_residuals)  # already in m

# Calculate Errors (With Q) - convert to m for plotting
error_pos_Q = np.linalg.norm(est_pos_Q - true_pos, axis=1) * 1000.0  # m
error_vel_Q = np.linalg.norm(est_vel_Q - true_vel, axis=1) * 1000.0  # m/s

# --- 5. PLOTTING COMPARISON (Task f & g) ---

# Comparison: Position Error (No Q vs With Q)
fig9, ax9 = plt.subplots(figsize=(10, 6))
ax9.plot(t, error_pos, 'b--', label='No Process Noise (Q=0)') # From Part 2
ax9.plot(t, error_pos_Q, 'g-', linewidth=2, label=f'With Process Noise ($\\sigma$={sigma_a})')
ax9.set_xlabel('Time [s]')
ax9.set_ylabel('Position Error [m]')
ax9.set_title('Effect of Process Noise on Position Error')
ax9.grid(True)
ax9.legend()
plt.savefig(path.join(path.dirname(__file__), 'plots/task_f_process_noise_comparison.png'), dpi=300, bbox_inches='tight')
print('Saved: plots/task_f_process_noise_comparison.png')

# RMS Residuals Plot
# RMS array length is same as epochs (N)
fig10, ax10 = plt.subplots(figsize=(10, 6))
ax10.plot(t, rms_history, 'k-', label='RMS Residuals')
ax10.set_xlabel('Time [s]')
ax10.set_ylabel('RMS Value [m]')
ax10.set_title('RMS of Measurement Residuals (With Process Noise)')
ax10.grid(True)
ax10.legend()
plt.savefig(path.join(path.dirname(__file__), 'plots/task_g_rms_residuals.png'), dpi=300, bbox_inches='tight')
print('Saved: plots/task_g_rms_residuals.png')

# Combined Plot (RMS vs Position Error)
fig11, ax11 = plt.subplots(figsize=(10, 6))
ax11.plot(t, rms_history, 'b-', label='RMS Residuals')
ax11.plot(t, error_pos_Q, 'r--', label=r'Position Error ($\delta r$)')
ax11.set_xlabel('Time [s]')
ax11.set_ylabel('Value [m]')
ax11.set_title('Task g: RMS Residuals vs Position Errors')
ax11.legend()
ax11.grid(True)
plt.savefig(path.join(path.dirname(__file__), 'plots/task_g_rms_vs_errors.png'), dpi=300, bbox_inches='tight')
print('Saved: plots/task_g_rms_vs_errors.png')

print('\n' + '='*60)
print('All plots saved to assignment3/plots/')
print('='*60)