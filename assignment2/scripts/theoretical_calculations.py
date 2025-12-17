"""
Theoretical Orbit Determination Calculations

This script calculates the answers to 3 theoretical questions about
GNSS positioning effects for a LEO satellite receiver.
"""

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================
c = 299792.458  # Speed of light [km/s]
omega_e = 7.292115e-5  # Earth rotation rate [rad/s]
mu_earth = 398600.4418  # Earth gravitational parameter [km^3/s^2]
Re = 6378.137  # Earth radius [km]

# ============================================================================
# SCENARIO PARAMETERS
# ============================================================================
# LEO satellite receiver
h_leo = 500  # Altitude [km]
r_leo = Re + h_leo  # Orbital radius [km]

# GPS satellites
a_gps = 26560  # Semi-major axis [km]
e_gps = 0.01  # Eccentricity

# Clock error
delta_t_clk = 0.1e-3  # Transmitter clock error [s] (0.1 ms)

print("="*70)
print("THEORETICAL ORBIT DETERMINATION CALCULATIONS")
print("="*70)
print("\nCONSTANTS:")
print(f"  Speed of light (c):           {c} km/s")
print(f"  Earth rotation rate (ω_e):    {omega_e} rad/s")
print(f"  Earth gravity (μ):            {mu_earth} km³/s²")
print(f"  Earth radius (R_e):           {Re} km")
print("\nSCENARIO:")
print(f"  LEO altitude:                 {h_leo} km")
print(f"  LEO orbital radius:           {r_leo} km")
print(f"  GPS semi-major axis:          {a_gps} km")
print(f"  GPS eccentricity:             {e_gps}")
print(f"  Transmitter clock error:      {delta_t_clk*1e3} ms")
print("="*70)

# ============================================================================
# QUESTION 1: CLOCK OFFSET EFFECT
# ============================================================================
print("\n" + "="*70)
print("QUESTION 1: CLOCK OFFSET EFFECT")
print("="*70)

# Range error due to clock error
Delta_rho_clock = c * delta_t_clk  # [km]

print(f"\nFormula: Δρ = c × δt")
print(f"\nCalculation:")
print(f"  Δρ = {c} km/s × {delta_t_clk*1e3} ms")
print(f"  Δρ = {c} km/s × {delta_t_clk} s")
print(f"  Δρ = {Delta_rho_clock} km")
print(f"\n>>> ANSWER: Range error = {Delta_rho_clock} km = {Delta_rho_clock*1000} m")

# ============================================================================
# QUESTION 2: LIGHT TIME (SAGNAC) EFFECT
# ============================================================================
print("\n" + "="*70)
print("QUESTION 2: LIGHT TIME (SAGNAC) EFFECT")
print("="*70)

# Calculate GPS satellite position range (min and max)
r_gps_perigee = a_gps * (1 - e_gps)  # Closest point
r_gps_apogee = a_gps * (1 + e_gps)   # Farthest point

# Geometry for LEO-GPS ranging
# Minimum range: GPS at zenith (directly above LEO)
# Use apogee distance for worst case
rho_min = r_gps_apogee - r_leo  # [km]

# Maximum range: GPS at horizon (tangent to LEO orbit)
# Geometric distance from LEO to horizon + GPS to horizon
tangent_leo = np.sqrt(r_leo**2 - Re**2)
tangent_gps = np.sqrt(r_gps_apogee**2 - Re**2)

rho_max = tangent_leo + tangent_gps

# Signal travel times
tau_min = rho_min / c  # [s]
tau_max = rho_max / c  # [s]

# Sagnac effect calculation
# Note: Lecture notes Section 6.2.1 refers to this as "Light time effect" 
# caused by motion "due to Earth rotation".
# The Sagnac correction accounts for Earth rotation during signal travel
# The magnitude is approximately: Δρ_sagnac ≈ ω_e * tau * r * sin(angle)
# For a more accurate calculation, the correction is:
# Δρ_sagnac = (ω_e / c) * (r_gps × r_leo)_z component projected
# Or approximately: ω_e * tau * (r_gps + r_leo) / 2 for the perpendicular component

# Conservative estimate using the cross-range displacement
# Maximum Sagnac occurs when GPS and LEO are perpendicular in the ECEF frame
# Δρ_sagnac ≈ ω_e * tau * r * sin(angle)

# For MINIMUM Sagnac (Zenith case):
# The geometry is nearly aligned along the radial direction (angle ≈ 0)
# sin(angle) ≈ 0, so Sagnac effect is minimal
# For practical purposes, even a small misalignment gives a small effect
angle_min = 0.0  # Radians (aligned geometry)
Delta_rho_sagnac_min = omega_e * tau_min * r_leo * np.sin(angle_min)  # [km]

# For MAXIMUM Sagnac (Horizon case):
# Maximum cross-range displacement occurs when geometry is perpendicular (angle ≈ 90°)
# sin(90°) = 1
angle_max = np.pi / 2  # 90 degrees in radians (perpendicular geometry)
Delta_rho_sagnac_max = omega_e * tau_max * r_gps_apogee * np.sin(angle_max)  # [km]

print(f"\nGeometry Analysis:")
print(f"  LEO orbital radius:           {r_leo:.3f} km")
print(f"  GPS perigee radius:           {r_gps_perigee:.3f} km")
print(f"  GPS apogee radius:            {r_gps_apogee:.3f} km")
print(f"\nRange Calculations:")
print(f"  Minimum range (Zenith):       {rho_min:.3f} km")
print(f"  Maximum range (Horizon):      {rho_max:.3f} km")
print(f"\nSignal Travel Time:")
print(f"  τ_min = {rho_min:.3f} / {c} = {tau_min*1e3:.6f} ms")
print(f"  τ_max = {rho_max:.3f} / {c} = {tau_max*1e3:.6f} ms")
print(f"\nSagnac Correction (ω_e × τ × r):")
print(f"  Δρ_sagnac_min ≈ {omega_e:.6e} × {tau_min:.6f} × {r_leo:.3f}")
print(f"  Δρ_sagnac_min ≈ {Delta_rho_sagnac_min*1e3:.3f} m")
print(f"\n  Δρ_sagnac_max ≈ {omega_e:.6e} × {tau_max:.6f} × {r_gps_apogee:.3f}")
print(f"  Δρ_sagnac_max ≈ {Delta_rho_sagnac_max*1e3:.3f} m")
print(f"\n>>> ANSWER:")
print(f"    Minimum Sagnac effect: {Delta_rho_sagnac_min*1e3:.3f} m")
print(f"    Maximum Sagnac effect: {Delta_rho_sagnac_max*1e3:.3f} m")

# ============================================================================
# QUESTION 3: RELATIVISTIC EFFECT (ECCENTRICITY)
# ============================================================================
print("\n" + "="*70)
print("QUESTION 3: RELATIVISTIC EFFECT (ECCENTRICITY)")
print("="*70)

# Relativistic effect formula: δt_rel = -2 * (r · v) / c^2
# For an elliptical orbit, the radial component of velocity is:
# v_r = dr/dt = (n * a * e * sin(f)) / sqrt(1 - e^2)
# where n is the mean motion, f is true anomaly
# 
# The dot product r · v equals r * v_r (radial component)
# Maximum occurs when sin(f) = ±1 (at f = 90° or 270°)

# Mean motion
n = np.sqrt(mu_earth / a_gps**3)  # [rad/s]

# At true anomaly f = 90° or 270°:
# r = a(1 - e^2) / (1 + e*cos(90°)) = a(1 - e^2)
r_at_90 = a_gps * (1 - e_gps**2)  # [km]

# Radial velocity at f = 90°:
# v_r = (n * a * e * sin(90°)) / sqrt(1 - e^2)
v_r_max = (n * a_gps * e_gps * 1.0) / np.sqrt(1 - e_gps**2)  # [km/s]

# Maximum r · v
r_dot_v_max = r_at_90 * v_r_max  # [km^2/s]

# Relativistic time effect
delta_t_rel_max = 2.0 * np.abs(r_dot_v_max) / (c**2)  # [s]

# Convert to range error
Delta_rho_rel_max = c * delta_t_rel_max  # [km]

# Minimum effect occurs at apogee and perigee where sin(f) = 0
Delta_rho_rel_min = 0.0  # [km]

print(f"\nOrbital Parameters:")
print(f"  Mean motion (n):              {n:.6e} rad/s")
print(f"  Radius at f=90°:              {r_at_90:.3f} km")
print(f"\nRadial Velocity Calculation:")
print(f"  v_r = (n × a × e) / √(1-e²)")
print(f"  v_r_max = ({n:.6e} × {a_gps} × {e_gps}) / √(1-{e_gps}²)")
print(f"  v_r_max = {v_r_max:.6f} km/s")
print(f"\nDot Product r·v:")
print(f"  (r·v)_max = r × v_r")
print(f"  (r·v)_max = {r_at_90:.3f} × {v_r_max:.6f}")
print(f"  (r·v)_max = {r_dot_v_max:.3f} km²/s")
print(f"\nRelativistic Time Effect:")
print(f"  δt_rel = 2 × |r·v| / c²")
print(f"  δt_rel_max = 2 × {r_dot_v_max:.3f} / ({c})²")
print(f"  δt_rel_max = {delta_t_rel_max:.6e} s = {delta_t_rel_max*1e9:.3f} ns")
print(f"\nRange Error:")
print(f"  Δρ_rel = c × δt_rel")
print(f"  Δρ_rel_max = {c} × {delta_t_rel_max:.6e}")
print(f"  Δρ_rel_max = {Delta_rho_rel_max*1e3:.6f} m")
print(f"\n>>> ANSWER:")
print(f"    Minimum relativistic effect: {Delta_rho_rel_min} m")
print(f"    Maximum relativistic effect: {Delta_rho_rel_max*1e3:.6f} m")

# ============================================================================
# SUMMARY OF ANSWERS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY OF ANSWERS")
print("="*70)
print(f"\n1. CLOCK OFFSET EFFECT:")
print(f"   Range error = {Delta_rho_clock:.2g} km")
print(f"\n2. LIGHT TIME (SAGNAC) EFFECT:")
print(f"   Minimum effect = {Delta_rho_sagnac_min*1e3:.2g} m")
print(f"   Maximum effect = {Delta_rho_sagnac_max*1e3:.2g} m")
print(f"\n3. RELATIVISTIC EFFECT (ECCENTRICITY):")
print(f"   Minimum effect = {Delta_rho_rel_min:.2g} m")
print(f"   Maximum effect = {Delta_rho_rel_max*1e3:.2g} m")
print("="*70)
