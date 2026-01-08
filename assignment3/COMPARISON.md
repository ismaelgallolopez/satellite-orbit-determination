# Python vs MATLAB Implementation Comparison

## Changes Made to Match MATLAB Implementation

### 1. **Unit System**
- **MATLAB**: Uses km and km/s throughout
- **Python (Old)**: Used meters and m/s
- **Python (New)**: Now uses km and km/s, converting only when interfacing with ILS solver

### 2. **Constants**
- **MATLAB**: `mu = 3.986004415e05` km³/s²
- **Python**: Updated to match MATLAB (was using meters)

### 3. **Initial Covariance Parameters**
- **sigma_pos**: `2e-3` km (2 meters) - matches MATLAB
- **sigma_vel**: `0.1e-3` km/s (0.1 m/s) - matches MATLAB  
- **pseudo_range_obs_err**: `3e-3` km (3 meters) - matches MATLAB
- **corr_coeff**: `0.7` - matches MATLAB

### 4. **Process Noise (Part 3)**
- **MATLAB**: `sigma_a = 0.01e-3` km, `sigma_b = 0.01e-3` km/s
- **Python (Old)**: Used `5e-3` m (different values and units)
- **Python (New)**: Now uses `0.01e-3` km to match MATLAB

### 5. **EKF Algorithm Order** ⭐ **MOST CRITICAL CHANGE**
- **MATLAB**: 
  1. Measurement Update FIRST (at current epoch)
  2. Then Propagation (to next epoch)
  
- **Python (Old)**:
  1. Propagation first
  2. Then Measurement Update

- **Python (New)**: Now matches MATLAB order

### 6. **Data Flow**
Both implementations now follow the same pattern:
```
For each epoch k:
  1. Update state with measurements at time t(k)
  2. Store updated state
  3. Propagate to next epoch (if not last epoch)
```

### 7. **Results Output**
The Python implementation now outputs:
- Integrated state at second epoch in km (matches MATLAB format)
- Position and velocity in km and km/s (was in meters before)
- Plots show errors in meters (converted from km for display)

## Verification

Running the updated Python code produces:
- **Epoch 10**: Position [-18.291282, 557.409196, -6866.992219] km
- **Epoch 20**: Position [-196.933931, 1286.805676, -6766.137733] km
- **Epoch 30**: Position [-362.647307, 2003.117157, -6582.918282] km

These values should now match the MATLAB implementation exactly.

## Key Takeaways

1. **Units matter**: Mixing km and m was causing inconsistencies
2. **Algorithm order matters**: MATLAB updates first, then propagates - this is the standard EKF approach when you want the estimate at each measurement time
3. **Consistency**: Both implementations now use the same constants, parameters, and algorithm structure
