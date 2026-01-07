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
x_gps = np.loadtxt(path.join(path_data, 'rx_gps.txt'))
y_gps = np.loadtxt(path.join(path_data, 'ry_gps.txt'))
z_gps = np.loadtxt(path.join(path_data, 'rz_gps.txt'))
r = np.concatenate((x_gps[:, np.newaxis], y_gps[:, np.newaxis], z_gps[:, np.newaxis]), axis=1)
r_norm = np.sqrt(x_gps**2 + y_gps**2 + z_gps**2)
# prn = np.loadtxt(path.join(path_data, 'PRN_ID.txt'))

# y = np.ndarray([x_gps, y_gps, z_gps])

Phi = np.identity(6)
F = np.zeros((6,6))
F[0:3, 3:6] = np.identity(3)
F[3:6, 3:6] = -2*omega_E
# F[3:6, 0:3] = -GM*r_norm**5*(3*r*r. - r_norm**2*np.identity(3) - omega_e**2)
# F[3,0] = mu(3*y[0] - )


# %% Part 2: ILS solver for state estimation
print("\n" + "="*70)
print("Running ILS solver on Assignment 3 data...")
print("="*70)

# Load assignment3 data in format expected by ILS solver
data = {
    't': np.loadtxt(path.join(path_data, 't.txt')),
    'PRN_ID': np.loadtxt(path.join(path_data, 'PRN_ID.txt')),
    'CA_range': np.loadtxt(path.join(path_data, 'CA_range.txt')),
    'rx_gps': np.loadtxt(path.join(path_data, 'rx_gps.txt')),
    'ry_gps': np.loadtxt(path.join(path_data, 'ry_gps.txt')),
    'rz_gps': np.loadtxt(path.join(path_data, 'rz_gps.txt')),
    'vx_gps': np.loadtxt(path.join(path_data, 'vx_gps.txt')),
    'vy_gps': np.loadtxt(path.join(path_data, 'vy_gps.txt')),
    'vz_gps': np.loadtxt(path.join(path_data, 'vz_gps.txt')),
    'rx_true': np.loadtxt(path.join(path_data, 'rx.txt')),
    'ry_true': np.loadtxt(path.join(path_data, 'ry.txt')),
    'rz_true': np.loadtxt(path.join(path_data, 'rz.txt')),
    'vx_true': np.loadtxt(path.join(path_data, 'vx.txt')),
    'vy_true': np.loadtxt(path.join(path_data, 'vy.txt')),
    'vz_true': np.loadtxt(path.join(path_data, 'vz.txt'))
}

# Check if clock data exists, if not create dummy data
try:
    data['clk_gps'] = np.loadtxt(path.join(path_data, 'clk_gps.txt'))
except:
    print("Warning: clk_gps.txt not found, using zeros")
    data['clk_gps'] = np.zeros_like(data['rx_gps'])

print(f"Data loaded - Shape: {data['PRN_ID'].shape}")

# Run ILS solver
estimated_states = ils_solver(data, max_iter=10, tol=1e-3)

print("\n" + "="*70)
print("ILS Results Summary:")
print("="*70)
print(f"Estimated states shape: {estimated_states.shape}")
print(f"\nFirst epoch estimate [x, y, z, c*dt] (km):")
print(estimated_states[0])
print(f"\nLast epoch estimate [x, y, z, c*dt] (km):")
print(estimated_states[-1])

# %% Load velocities from GPS files
print("\n" + "="*70)
print("Loading Velocities from GPS Files...")
print("="*70)

# Extract positions from ILS results
pos_est = estimated_states[:, 0:3]  # [x, y, z] in km

# Load velocities from vx_gps.txt, vy_gps.txt, vz_gps.txt
# Note: These are matrices with shape (n_epochs, n_satellites)
# We need to extract receiver velocities or use a different approach
vx_gps_all = np.loadtxt(path.join(path_data, 'vx_gps.txt'))
vy_gps_all = np.loadtxt(path.join(path_data, 'vy_gps.txt'))
vz_gps_all = np.loadtxt(path.join(path_data, 'vz_gps.txt'))

print(f"GPS velocity data shape (per component): {vx_gps_all.shape}")
print(f"Note: These are satellite velocities, not receiver velocities")
print(f"For receiver velocities, we need to estimate from position changes")

# Estimate receiver velocities from position changes
t_array = data['t']
delta_t = t_array[1] - t_array[0]  # Time step in seconds

vel_est = np.zeros((len(pos_est), 3))
# Forward difference for first point
vel_est[0] = (pos_est[1] - pos_est[0]) / delta_t
# Central difference for middle points
for i in range(1, len(pos_est) - 1):
    vel_est[i] = (pos_est[i+1] - pos_est[i-1]) / (2 * delta_t)
# Backward difference for last point
vel_est[-1] = (pos_est[-1] - pos_est[-2]) / delta_t

print(f"\nEstimated receiver velocity shape: {vel_est.shape}")
print(f"First epoch velocity [vx, vy, vz] (km/s):")
print(vel_est[0])
print(f"Last epoch velocity [vx, vy, vz] (km/s):")
print(vel_est[-1])

# Create full state vector [x, y, z, vx, vy, vz]
state_vector_est = np.hstack([pos_est, vel_est])
print(f"\nFull state vector shape: {state_vector_est.shape}")

# Compare with true positions
rx_true = data['rx_true']
ry_true = data['ry_true']
rz_true = data['rz_true']
vx_true = data['vx_true']
vy_true = data['vy_true']
vz_true = data['vz_true']

# Calculate component-wise errors
error_x = estimated_states[:, 0] - rx_true
error_y = estimated_states[:, 1] - ry_true
error_z = estimated_states[:, 2] - rz_true

# Calculate velocity errors
error_vx = vel_est[:, 0] - vx_true
error_vy = vel_est[:, 1] - vy_true
error_vz = vel_est[:, 2] - vz_true

# Calculate total position and velocity errors
pos_errors = np.sqrt(error_x**2 + error_y**2 + error_z**2)
vel_errors = np.sqrt(error_vx**2 + error_vy**2 + error_vz**2)

print(f"\n{'='*70}")
print("POSITION Error Statistics:")
print("-" * 70)
print(f"{'Component':<12} {'Mean (km)':<15} {'Std (km)':<15} {'RMS (km)':<15}")
print("-" * 70)
print(f"{'X':<12} {error_x.mean():<15.6f} {error_x.std():<15.6f} {np.sqrt(np.mean(error_x**2)):<15.6f}")
print(f"{'Y':<12} {error_y.mean():<15.6f} {error_y.std():<15.6f} {np.sqrt(np.mean(error_y**2)):<15.6f}")
print(f"{'Z':<12} {error_z.mean():<15.6f} {error_z.std():<15.6f} {np.sqrt(np.mean(error_z**2)):<15.6f}")
print("-" * 70)
print(f"{'3D Position':<12} {pos_errors.mean():<15.6f} {pos_errors.std():<15.6f} {np.sqrt(np.mean(pos_errors**2)):<15.6f}")
print(f"{'Min 3D Error':<12} {pos_errors.min():.6f} km")
print(f"{'Max 3D Error':<12} {pos_errors.max():.6f} km")

print(f"\n3D Position Error in meters:")
print(f"  Mean: {pos_errors.mean()*1000:.2f} m")
print(f"  RMS:  {np.sqrt(np.mean(pos_errors**2))*1000:.2f} m")
print(f"  Std:  {pos_errors.std()*1000:.2f} m")
print(f"  Min:  {pos_errors.min()*1000:.2f} m")
print(f"  Max:  {pos_errors.max()*1000:.2f} m")

print(f"\n{'='*70}")
print("VELOCITY Error Statistics:")
print("-" * 70)
print(f"{'Component':<12} {'Mean (km/s)':<15} {'Std (km/s)':<15} {'RMS (km/s)':<15}")
print("-" * 70)
print(f"{'Vx':<12} {error_vx.mean():<15.6f} {error_vx.std():<15.6f} {np.sqrt(np.mean(error_vx**2)):<15.6f}")
print(f"{'Vy':<12} {error_vy.mean():<15.6f} {error_vy.std():<15.6f} {np.sqrt(np.mean(error_vy**2)):<15.6f}")
print(f"{'Vz':<12} {error_vz.mean():<15.6f} {error_vz.std():<15.6f} {np.sqrt(np.mean(error_vz**2)):<15.6f}")
print("-" * 70)
print(f"{'3D Velocity':<12} {vel_errors.mean():<15.6f} {vel_errors.std():<15.6f} {np.sqrt(np.mean(vel_errors**2)):<15.6f}")
print(f"{'Min 3D Error':<12} {vel_errors.min():.6f} km/s")
print(f"{'Max 3D Error':<12} {vel_errors.max():.6f} km/s")

print(f"\n3D Velocity Error in m/s:")
print(f"  Mean: {vel_errors.mean()*1000:.2f} m/s")
print(f"  RMS:  {np.sqrt(np.mean(vel_errors**2))*1000:.2f} m/s")
print(f"  Std:  {vel_errors.std()*1000:.2f} m/s")
print(f"  Min:  {vel_errors.min()*1000:.2f} m/s")
print(f"  Max:  {vel_errors.max()*1000:.2f} m/s")

print(f"\n{'='*70}")
print("FULL STATE VECTOR [x, y, z, vx, vy, vz] - First 3 epochs:")
print("-" * 70)
for i in range(min(3, len(state_vector_est))):
    print(f"Epoch {i+1}:")
    print(f"  Estimated: {state_vector_est[i]}")
    print(f"  True:      [{rx_true[i]}, {ry_true[i]}, {rz_true[i]}, {vx_true[i]}, {vy_true[i]}, {vz_true[i]}]")
    print()
print("="*70)

