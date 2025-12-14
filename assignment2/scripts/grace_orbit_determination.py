"""
Iterative Least-Squares (ILS) Orbit Determination for GRACE-FO Satellite
Using GPS Pseudo-range Measurements

This script implements:
- Basic model (clock corrections only)
- Advanced model (with Light-Time and Relativistic corrections)
- Error analysis with velocity correction hint
- Residual analysis
- PDOP calculation and plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Physical constants
c = 299792.458  # Speed of light [km/s]
omega_e = 7.2921151467e-5  # Earth rotation rate [rad/s]

# Data directory (relative to project root)
data_dir = Path(__file__).parent.parent / 'data' / 'provided_assignment'


def load_grace_data():
    """Load all GRACE-FO orbit determination data from text files."""
    print("Loading GRACE-FO data...")
    
    # Load time data
    t = np.loadtxt(data_dir / 't.txt')  # Time [s]
    
    # Load PRN IDs
    PRN_ID = np.loadtxt(data_dir / 'PRN_ID.txt')  # Satellite PRN IDs
    
    # Load GPS satellite positions (ECEF) [km]
    x_GPS = np.loadtxt(data_dir / 'rx_gps.txt')
    y_GPS = np.loadtxt(data_dir / 'ry_gps.txt')
    z_GPS = np.loadtxt(data_dir / 'rz_gps.txt')
    
    # Load GPS satellite velocities (ECEF) [km/s]
    vx_GPS = np.loadtxt(data_dir / 'vx_gps.txt')
    vy_GPS = np.loadtxt(data_dir / 'vy_gps.txt')
    vz_GPS = np.loadtxt(data_dir / 'vz_gps.txt')
    
    # Load GPS satellite clock offsets [s]
    dt_GPS = np.loadtxt(data_dir / 'clk_gps.txt')
    
    # Load observed pseudo-ranges [km]
    rho_obs = np.loadtxt(data_dir / 'CA_range.txt')
    
    # Load GRACE-FO precise reference positions (ECEF) [km]
    x_GRACE = np.loadtxt(data_dir / 'rx.txt')
    y_GRACE = np.loadtxt(data_dir / 'ry.txt')
    z_GRACE = np.loadtxt(data_dir / 'rz.txt')
    
    # Load GRACE-FO precise velocities (ECEF) [km/s]
    vx_GRACE = np.loadtxt(data_dir / 'vx.txt')
    vy_GRACE = np.loadtxt(data_dir / 'vy.txt')
    vz_GRACE = np.loadtxt(data_dir / 'vz.txt')
    
    print(f"Data loaded successfully!")
    print(f"Number of epochs: {len(t)}")
    print(f"Data shape: {PRN_ID.shape}")
    
    return {
        't': t,
        'PRN_ID': PRN_ID,
        'x_GPS': x_GPS,
        'y_GPS': y_GPS,
        'z_GPS': z_GPS,
        'vx_GPS': vx_GPS,
        'vy_GPS': vy_GPS,
        'vz_GPS': vz_GPS,
        'dt_GPS': dt_GPS,
        'rho_obs': rho_obs,
        'x_GRACE': x_GRACE,
        'y_GRACE': y_GRACE,
        'z_GRACE': z_GRACE,
        'vx_GRACE': vx_GRACE,
        'vy_GRACE': vy_GRACE,
        'vz_GRACE': vz_GRACE
    }


def rotation_matrix_z(angle):
    """
    Create a rotation matrix around the z-axis for Earth rotation correction.
    
    Parameters:
    -----------
    angle : float
        Rotation angle in radians
        
    Returns:
    --------
    R : ndarray (3x3)
        Rotation matrix
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    R = np.array([
        [cos_a, sin_a, 0],
        [-sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    return R


def light_time_correction(r_GPS, r_rx_est):
    """
    Apply light-time (Sagnac) correction for Earth rotation during signal travel.
    
    Parameters:
    -----------
    r_GPS : ndarray (3,)
        GPS satellite position [km]
    r_rx_est : ndarray (3,)
        Estimated receiver position [km]
        
    Returns:
    --------
    r_GPS_corr : ndarray (3,)
        Corrected GPS position [km]
    """
    # Calculate geometric range
    rho_geo = np.linalg.norm(r_GPS - r_rx_est)
    
    # Signal travel time
    tau = rho_geo / c
    
    # Rotation angle due to Earth rotation
    angle = -omega_e * tau
    
    # Apply rotation to GPS position
    R_z = rotation_matrix_z(angle)
    r_GPS_corr = R_z @ r_GPS
    
    return r_GPS_corr


def relativistic_correction(r_GPS, v_GPS):
    """
    Calculate relativistic correction due to GPS orbit eccentricity.
    
    Parameters:
    -----------
    r_GPS : ndarray (3,)
        GPS satellite position [km]
    v_GPS : ndarray (3,)
        GPS satellite velocity [km/s]
        
    Returns:
    --------
    delta_t_rel : float
        Relativistic time correction [s]
    """
    # Relativistic effect: delta_t = -2 * (r · v) / c^2
    r_dot_v = np.dot(r_GPS, v_GPS)
    delta_t_rel = -2.0 * r_dot_v / (c**2)
    
    return delta_t_rel


def run_orbit_determination(data, corrections=False, max_iter=10, tol=1e-6):
    """
    Run Iterative Least-Squares orbit determination.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all loaded data
    corrections : bool
        If True, apply Light-Time and Relativistic corrections
    max_iter : int
        Maximum number of ILS iterations
    tol : float
        Convergence tolerance [km]
        
    Returns:
    --------
    results : dict
        Dictionary containing estimated states, errors, residuals, etc.
    """
    # Extract data
    t = data['t']
    PRN_ID = data['PRN_ID']
    x_GPS = data['x_GPS']
    y_GPS = data['y_GPS']
    z_GPS = data['z_GPS']
    vx_GPS = data['vx_GPS']
    vy_GPS = data['vy_GPS']
    vz_GPS = data['vz_GPS']
    dt_GPS = data['dt_GPS']
    rho_obs = data['rho_obs']
    x_GRACE = data['x_GRACE']
    y_GRACE = data['y_GRACE']
    z_GRACE = data['z_GRACE']
    vx_GRACE = data['vx_GRACE']
    vy_GRACE = data['vy_GRACE']
    vz_GRACE = data['vz_GRACE']
    
    N_epochs = len(t)
    
    # Initialize result arrays
    x_est = np.zeros(N_epochs)
    y_est = np.zeros(N_epochs)
    z_est = np.zeros(N_epochs)
    dt_r_est = np.zeros(N_epochs)  # Receiver clock offset [s]
    residuals = []  # Store residuals for each epoch
    PDOP = np.zeros(N_epochs)
    
    model_name = "Advanced" if corrections else "Basic"
    print(f"\nRunning {model_name} Model ILS orbit determination...")
    
    # Loop through all epochs
    for i in range(N_epochs):
        if i % 50 == 0:
            print(f"Processing epoch {i+1}/{N_epochs}...")
        
        # Get valid satellites for this epoch (non-zero PRN_ID and rho_obs)
        valid_mask = (PRN_ID[i] != 0) & (rho_obs[i] != 0)
        
        if not np.any(valid_mask):
            # No valid observations, use previous estimate or NaN
            if i > 0:
                x_est[i] = x_est[i-1]
                y_est[i] = y_est[i-1]
                z_est[i] = z_est[i-1]
                dt_r_est[i] = dt_r_est[i-1]
            else:
                x_est[i] = np.nan
                y_est[i] = np.nan
                z_est[i] = np.nan
                dt_r_est[i] = np.nan
            PDOP[i] = np.nan
            residuals.append(np.array([]))
            continue
        
        # Filter data for valid satellites
        r_GPS_valid = np.column_stack([
            x_GPS[i][valid_mask],
            y_GPS[i][valid_mask],
            z_GPS[i][valid_mask]
        ])
        v_GPS_valid = np.column_stack([
            vx_GPS[i][valid_mask],
            vy_GPS[i][valid_mask],
            vz_GPS[i][valid_mask]
        ])
        dt_GPS_valid = dt_GPS[i][valid_mask]
        rho_obs_valid = rho_obs[i][valid_mask]
        
        N_valid = len(rho_obs_valid)
        
        # Initial guess: center of Earth or previous estimate
        if i == 0:
            x_state = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, z, c*dt_r]
        else:
            x_state = np.array([x_est[i-1], y_est[i-1], z_est[i-1], c*dt_r_est[i-1]])
        
        # Iterative Least Squares
        for iteration in range(max_iter):
            # Current estimate
            r_rx = x_state[0:3]
            c_dt_r = x_state[3]
            dt_r = c_dt_r / c
            
            # Initialize arrays for this iteration
            rho_calc = np.zeros(N_valid)
            H = np.zeros((N_valid, 4))
            
            # Loop through valid satellites
            for j in range(N_valid):
                r_GPS_j = r_GPS_valid[j]
                v_GPS_j = v_GPS_valid[j]
                dt_GPS_j = dt_GPS_valid[j]
                
                # Apply corrections if enabled
                if corrections:
                    # Light-time correction
                    r_GPS_j = light_time_correction(r_GPS_j, r_rx)
                    
                    # Relativistic correction
                    delta_t_rel = relativistic_correction(r_GPS_valid[j], v_GPS_j)
                else:
                    delta_t_rel = 0.0
                
                # Calculate geometric range
                dr = r_GPS_j - r_rx
                rho_geo = np.linalg.norm(dr)
                
                # Calculate modeled pseudo-range
                rho_calc[j] = rho_geo + c * (dt_r - dt_GPS_j - delta_t_rel)
                
                # Design matrix row
                H[j, 0:3] = -dr / rho_geo  # Partial derivatives w.r.t. position
                H[j, 3] = 1.0  # Partial derivative w.r.t. c*dt_r
            
            # Residual vector
            dy = rho_obs_valid - rho_calc
            
            # Solve normal equations: dx = (H^T H)^-1 H^T dy
            HTH = H.T @ H
            HTdy = H.T @ dy
            
            try:
                dx = np.linalg.solve(HTH, HTdy)
            except np.linalg.LinAlgError:
                # Singular matrix, use pseudoinverse
                dx = np.linalg.lstsq(HTH, HTdy, rcond=None)[0]
            
            # Update state
            x_state = x_state + dx
            
            # Check convergence
            if np.linalg.norm(dx[0:3]) < tol:
                break
        
        # Store results
        x_est[i] = x_state[0]
        y_est[i] = x_state[1]
        z_est[i] = x_state[2]
        dt_r_est[i] = x_state[3] / c
        residuals.append(dy)
        
        # Calculate PDOP
        # PDOP = sqrt(Q_11 + Q_22 + Q_33) where Q = (H^T H)^-1
        try:
            Q = np.linalg.inv(HTH)
            PDOP[i] = np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])
        except np.linalg.LinAlgError:
            PDOP[i] = np.nan
    
    print(f"{model_name} Model ILS completed!")
    
    # Calculate errors with velocity correction hint
    # x_ref_corr = x_ref + v_ref * dt_r
    x_ref_corr = x_GRACE + vx_GRACE * dt_r_est
    y_ref_corr = y_GRACE + vy_GRACE * dt_r_est
    z_ref_corr = z_GRACE + vz_GRACE * dt_r_est
    
    # Absolute position error (L2 norm)
    pos_error = np.sqrt(
        (x_est - x_ref_corr)**2 + 
        (y_est - y_ref_corr)**2 + 
        (z_est - z_ref_corr)**2
    )
    
    return {
        't': t,
        'x_est': x_est,
        'y_est': y_est,
        'z_est': z_est,
        'dt_r_est': dt_r_est,
        'x_ref_corr': x_ref_corr,
        'y_ref_corr': y_ref_corr,
        'z_ref_corr': z_ref_corr,
        'pos_error': pos_error,
        'residuals': residuals,
        'PDOP': PDOP
    }


def print_first_epochs_table(results, model_name, n_epochs=4):
    """Print table of estimated positions for first n epochs."""
    print(f"\n{model_name} Model - Estimated Positions (First {n_epochs} Epochs)")
    print("="*80)
    print(f"{'Epoch':<8} {'X (km)':<15} {'Y (km)':<15} {'Z (km)':<15} {'Error (m)':<12}")
    print("-"*80)
    
    for i in range(min(n_epochs, len(results['t']))):
        x = results['x_est'][i]
        y = results['y_est'][i]
        z = results['z_est'][i]
        err = results['pos_error'][i] * 1000  # Convert to meters
        
        # Print with cm precision (5 decimal places for km = cm)
        print(f"{i+1:<8} {x:>14.5f}  {y:>14.5f}  {z:>14.5f}  {err:>11.3f}")
    
    print("="*80)


def plot_position_error(results_basic, results_advanced):
    """Plot absolute position error vs time for both models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    t = results_basic['t']
    
    ax.plot(t, results_basic['pos_error'] * 1000, 'b-', 
            label='Basic Model', linewidth=1.5, alpha=0.7)
    ax.plot(t, results_advanced['pos_error'] * 1000, 'r-', 
            label='Advanced Model', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Absolute Position Error [m]', fontsize=12)
    ax.set_title('GRACE-FO Orbit Determination Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_residuals(results_basic, results_advanced):
    """Plot measurement residuals for both models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    t = results_basic['t']
    
    # Flatten residuals (convert list of arrays to single array per epoch)
    res_basic = []
    res_advanced = []
    t_res = []
    
    for i, (rb, ra) in enumerate(zip(results_basic['residuals'], 
                                      results_advanced['residuals'])):
        if len(rb) > 0:
            for r in rb:
                res_basic.append(r * 1000)  # Convert to meters
                t_res.append(t[i])
        if len(ra) > 0:
            for r in ra:
                res_advanced.append(r * 1000)  # Convert to meters
    
    # Plot as scatter or line
    ax.scatter(t_res, res_basic, c='blue', s=1, alpha=0.3, label='Basic Model')
    ax.scatter(t_res, res_advanced, c='red', s=1, alpha=0.3, label='Advanced Model')
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Residuals [m]', fontsize=12)
    ax.set_title('Pseudo-range Measurement Residuals', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    return fig


def plot_clock_offset(results):
    """Plot receiver clock offset vs time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    t = results['t']
    dt_r = results['dt_r_est'] * 1e6  # Convert to microseconds
    
    ax.plot(t, dt_r, 'g-', linewidth=1.5)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Receiver Clock Offset [μs]', fontsize=12)
    ax.set_title('GRACE-FO Receiver Clock Offset (Advanced Model)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pdop_and_error(results):
    """Plot PDOP and position error on dual y-axis."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    t = results['t']
    
    # Plot PDOP on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('PDOP', fontsize=12, color=color1)
    ax1.plot(t, results['PDOP'], color=color1, linewidth=1.5, label='PDOP')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot position error on right y-axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Position Error [m]', fontsize=12, color=color2)
    ax2.plot(t, results['pos_error'] * 1000, color=color2, linewidth=1.5, 
             label='Position Error', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title('PDOP and Position Error (Advanced Model)', 
                  fontsize=14, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    # Load data
    data = load_grace_data()
    
    # Run Basic Model (no corrections)
    print("\n" + "="*80)
    print("BASIC MODEL (Clock Corrections Only)")
    print("="*80)
    results_basic = run_orbit_determination(data, corrections=False)
    print_first_epochs_table(results_basic, "Basic")
    
    # Run Advanced Model (with corrections)
    print("\n" + "="*80)
    print("ADVANCED MODEL (with Light-Time and Relativistic Corrections)")
    print("="*80)
    results_advanced = run_orbit_determination(data, corrections=True)
    print_first_epochs_table(results_advanced, "Advanced")
    
    # Create plots directory if it doesn't exist
    plots_dir = Path(__file__).parent.parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot 1: Position Error Comparison
    fig1 = plot_position_error(results_basic, results_advanced)
    fig1.savefig(plots_dir / 'grace_position_error.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {plots_dir / 'grace_position_error.png'}")
    
    # Plot 2: Residuals Comparison
    fig2 = plot_residuals(results_basic, results_advanced)
    fig2.savefig(plots_dir / 'grace_residuals.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {plots_dir / 'grace_residuals.png'}")
    
    # Plot 3: Receiver Clock Offset
    fig3 = plot_clock_offset(results_advanced)
    fig3.savefig(plots_dir / 'grace_clock_offset.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {plots_dir / 'grace_clock_offset.png'}")
    
    # Plot 4: PDOP and Error
    fig4 = plot_pdop_and_error(results_advanced)
    fig4.savefig(plots_dir / 'grace_pdop_error.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {plots_dir / 'grace_pdop_error.png'}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    valid_basic = ~np.isnan(results_basic['pos_error'])
    valid_advanced = ~np.isnan(results_advanced['pos_error'])
    
    print(f"\nBasic Model:")
    print(f"  Mean Error:   {np.nanmean(results_basic['pos_error'][valid_basic])*1000:.3f} m")
    print(f"  RMS Error:    {np.sqrt(np.nanmean(results_basic['pos_error'][valid_basic]**2))*1000:.3f} m")
    print(f"  Max Error:    {np.nanmax(results_basic['pos_error'][valid_basic])*1000:.3f} m")
    
    print(f"\nAdvanced Model:")
    print(f"  Mean Error:   {np.nanmean(results_advanced['pos_error'][valid_advanced])*1000:.3f} m")
    print(f"  RMS Error:    {np.sqrt(np.nanmean(results_advanced['pos_error'][valid_advanced]**2))*1000:.3f} m")
    print(f"  Max Error:    {np.nanmax(results_advanced['pos_error'][valid_advanced])*1000:.3f} m")
    
    print(f"\nMean PDOP:      {np.nanmean(results_advanced['PDOP']):.3f}")
    print("="*80)


if __name__ == "__main__":
    main()
