"""
GNSS Point Positioning using Iterative Least Squares (ILS)

This script implements an ILS solver for GNSS positioning using L1/L2 data.
It estimates the receiver's state vector [x_r, y_r, z_r, c*dt_r] at each epoch.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Physical constants
c = 299792.458  # Speed of light [km/s]
omega_e = 7.2921151467e-5  # Earth rotation rate [rad/s]

# Data directory (relative to project root)
data_dir = Path(__file__).parent.parent / 'data' / 'provided_assignment'


def load_data():
    """Load all GNSS data from text files."""
    print("Loading data...")
    
    # Load time data
    t = np.loadtxt(data_dir / 't.txt')  # Time [s]
    
    # Load PRN IDs and CA range
    PRN_ID = np.loadtxt(data_dir / 'PRN_ID.txt')  # Satellite PRN IDs
    CA_range = np.loadtxt(data_dir / 'CA_range.txt')  # Pseudorange measurements [km]
    
    # Load GPS satellite positions (ECEF) [km]
    rx_gps = np.loadtxt(data_dir / 'rx_gps.txt')
    ry_gps = np.loadtxt(data_dir / 'ry_gps.txt')
    rz_gps = np.loadtxt(data_dir / 'rz_gps.txt')
    
    # Load GPS satellite velocities (ECEF) [km/s]
    vx_gps = np.loadtxt(data_dir / 'vx_gps.txt')
    vy_gps = np.loadtxt(data_dir / 'vy_gps.txt')
    vz_gps = np.loadtxt(data_dir / 'vz_gps.txt')
    
    # Load satellite clock corrections [s]
    clk_gps = np.loadtxt(data_dir / 'clk_gps.txt')
    
    # Load true receiver positions (for comparison) [km]
    rx_true = np.loadtxt(data_dir / 'rx.txt')
    ry_true = np.loadtxt(data_dir / 'ry.txt')
    rz_true = np.loadtxt(data_dir / 'rz.txt')
    
    # Load true receiver velocities [km/s]
    vx_true = np.loadtxt(data_dir / 'vx.txt')
    vy_true = np.loadtxt(data_dir / 'vy.txt')
    vz_true = np.loadtxt(data_dir / 'vz.txt')
    
    print(f"Data loaded successfully!")
    print(f"Number of epochs: {len(t)}")
    print(f"Data shape: {PRN_ID.shape}")
    
    return {
        't': t,
        'PRN_ID': PRN_ID,
        'CA_range': CA_range,
        'rx_gps': rx_gps,
        'ry_gps': ry_gps,
        'rz_gps': rz_gps,
        'vx_gps': vx_gps,
        'vy_gps': vy_gps,
        'vz_gps': vz_gps,
        'clk_gps': clk_gps,
        'rx_true': rx_true,
        'ry_true': ry_true,
        'rz_true': rz_true,
        'vx_true': vx_true,
        'vy_true': vy_true,
        'vz_true': vz_true
    }


def rotation_matrix_z(angle):
    """
    Create a rotation matrix around the z-axis.
    
    Parameters:
    -----------
    angle : float or ndarray
        Rotation angle in radians
        
    Returns:
    --------
    R : ndarray
        Rotation matrix (3x3) or array of rotation matrices
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    if np.isscalar(angle):
        R = np.array([
            [cos_a, sin_a, 0],
            [-sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    else:
        # Handle multiple angles at once
        n = len(angle)
        R = np.zeros((n, 3, 3))
        R[:, 0, 0] = cos_a
        R[:, 0, 1] = sin_a
        R[:, 1, 0] = -sin_a
        R[:, 1, 1] = cos_a
        R[:, 2, 2] = 1
    
    return R


def ils_solver(data, max_iter=10, tol=1e-3):
    """
    Iterative Least Squares solver for GNSS point positioning.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all loaded GNSS data
    max_iter : int
        Maximum number of iterations for the ILS
    tol : float
        Convergence tolerance [km]
        
    Returns:
    --------
    results : ndarray
        Estimated receiver states [x_r, y_r, z_r, c*dt_r] for each epoch
    """
    # Extract data
    t = data['t']
    PRN_ID = data['PRN_ID']
    CA_range = data['CA_range']
    rx_gps = data['rx_gps']
    ry_gps = data['ry_gps']
    rz_gps = data['rz_gps']
    vx_gps = data['vx_gps']
    vy_gps = data['vy_gps']
    vz_gps = data['vz_gps']
    clk_gps = data['clk_gps']
    rx_true = data['rx_true']
    ry_true = data['ry_true']
    rz_true = data['rz_true']
    
    N_epochs = len(t)
    
    # Initialize results array
    results = np.zeros((N_epochs, 4))
    
    # Initial guess for the first epoch
    x_est = np.array([rx_true[0], ry_true[0], rz_true[0], 0.0])
    
    print("\nStarting ILS processing...")
    
    # Loop through all epochs
    for i in range(N_epochs):
        if i % 100 == 0:
            print(f"Processing epoch {i+1}/{N_epochs}...")
        
        # 1. Select valid satellites (non-zero PRN_ID and non-zero CA_range)
        valid_mask = (PRN_ID[i] != 0) & (CA_range[i] != 0)
        
        # Filter data for valid satellites
        prn_valid = PRN_ID[i][valid_mask]
        rho_obs = CA_range[i][valid_mask]  # Observed pseudorange
        r_gps = np.column_stack([
            rx_gps[i][valid_mask],
            ry_gps[i][valid_mask],
            rz_gps[i][valid_mask]
        ])
        v_gps = np.column_stack([
            vx_gps[i][valid_mask],
            vy_gps[i][valid_mask],
            vz_gps[i][valid_mask]
        ])
        dt_clk = clk_gps[i][valid_mask]  # Satellite clock correction [s]
        
        N_valid = len(prn_valid)
        
        if N_valid < 4:
            print(f"Warning: Only {N_valid} valid satellites at epoch {i}")
            results[i] = x_est
            continue
        
        # 2. Iterative Least Squares
        for iteration in range(max_iter):
            # Current receiver position estimate
            r_rx = x_est[0:3]
            dt_rx = x_est[3] / c  # Receiver clock offset [s]
            
            # a. Calculate geometric range
            dr = r_gps - r_rx  # Vector from receiver to satellite
            rho_geo = np.linalg.norm(dr, axis=1)  # Geometric distance [km]
            
            # b. Light-time (Sagnac) correction
            tau = rho_geo / c  # Signal travel time [s]
            angle = -omega_e * tau  # Rotation angle [rad]
            
            # Apply rotation to each satellite position
            r_corr = np.zeros_like(r_gps)
            for j in range(N_valid):
                R_z = rotation_matrix_z(angle[j])
                r_corr[j] = R_z @ r_gps[j]
            
            # Recalculate geometric range with corrected positions
            dr_corr = r_corr - r_rx
            rho_geo = np.linalg.norm(dr_corr, axis=1)
            
            # c. Relativistic correction
            # delta_rel = -2 * (r_gps · v_gps) / c^2
            r_dot_v = np.sum(r_gps * v_gps, axis=1)
            delta_rel = -2.0 * r_dot_v / (c**2)  # [s]
            
            # d. Modeled range
            # rho_model = rho_geo + c * (dt_rx - dt_clk - delta_rel)
            rho_model = rho_geo + c * (dt_rx - dt_clk - delta_rel)
            
            # e. Design matrix H
            H = np.zeros((N_valid, 4))
            # First three columns: unit line-of-sight vector (negative)
            H[:, 0:3] = -dr_corr / rho_geo[:, np.newaxis]
            # Last column: all ones (for clock offset)
            H[:, 3] = 1.0
            
            # f. Residual vector
            dy = rho_obs - rho_model
            
            # g. Solve normal equations: dx = (H^T H)^-1 H^T dy
            HTH = H.T @ H
            HTdy = H.T @ dy
            dx = np.linalg.solve(HTH, HTdy)
            
            # h. Update state
            x_est = x_est + dx
            
            # i. Check convergence
            if np.linalg.norm(dx) < tol:
                break
        
        # Store result for this epoch
        results[i] = x_est
        
        # Use current result as initial guess for next epoch
        # (already in x_est, no need to reassign)
    
    print("ILS processing completed!")
    return results


def plot_results(data, results):
    """
    Plot the estimated receiver position versus true position.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all loaded GNSS data
    results : ndarray
        Estimated receiver states from ILS
    """
    t = data['t']
    rx_true = data['rx_true']
    ry_true = data['ry_true']
    rz_true = data['rz_true']
    
    # Extract estimated positions
    rx_est = results[:, 0]
    ry_est = results[:, 1]
    rz_est = results[:, 2]
    c_dt_r = results[:, 3]
    
    # Calculate position errors
    error_x = rx_est - rx_true
    error_y = ry_est - ry_true
    error_z = rz_est - rz_true
    error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: X-position
    axes[0, 0].plot(t, rx_true, 'b-', label='True', linewidth=2)
    axes[0, 0].plot(t, rx_est, 'r--', label='Estimated', linewidth=1.5)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('X Position [km]')
    axes[0, 0].set_title('Receiver X-Position')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Y-position
    axes[0, 1].plot(t, ry_true, 'b-', label='True', linewidth=2)
    axes[0, 1].plot(t, ry_est, 'r--', label='Estimated', linewidth=1.5)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Y Position [km]')
    axes[0, 1].set_title('Receiver Y-Position')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Z-position
    axes[1, 0].plot(t, rz_true, 'b-', label='True', linewidth=2)
    axes[1, 0].plot(t, rz_est, 'r--', label='Estimated', linewidth=1.5)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Z Position [km]')
    axes[1, 0].set_title('Receiver Z-Position')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: 3D position error
    axes[1, 1].plot(t, error_3d * 1000, 'g-', linewidth=2)  # Convert to meters
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('3D Position Error [m]')
    axes[1, 1].set_title('3D Position Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to plots directory
    plot_path = Path(__file__).parent.parent / 'plots' / 'gnss_positioning_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{plot_path}'")
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("POSITIONING STATISTICS")
    print("="*60)
    print(f"Mean 3D Error: {np.mean(error_3d)*1000:.3f} m")
    print(f"RMS 3D Error:  {np.sqrt(np.mean(error_3d**2))*1000:.3f} m")
    print(f"Max 3D Error:  {np.max(error_3d)*1000:.3f} m")
    print(f"Min 3D Error:  {np.min(error_3d)*1000:.3f} m")
    print(f"Std 3D Error:  {np.std(error_3d)*1000:.3f} m")
    print("="*60)
    print(f"Mean X Error: {np.mean(error_x)*1000:.3f} m")
    print(f"Mean Y Error: {np.mean(error_y)*1000:.3f} m")
    print(f"Mean Z Error: {np.mean(error_z)*1000:.3f} m")
    print("="*60)
    print(f"Mean Clock Offset: {np.mean(c_dt_r):.3f} km = {np.mean(c_dt_r)/c*1e9:.3f} ns")
    print("="*60)


def main():
    """Main execution function."""
    # Load data
    data = load_data()
    
    # Run ILS solver
    results = ils_solver(data, max_iter=10, tol=1e-3)
    
    # Plot results
    plot_results(data, results)


if __name__ == "__main__":
    main()
