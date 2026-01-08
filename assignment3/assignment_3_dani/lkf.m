function [x_est, P_est] = lkf(t, x0, P0, sigma_rho, CA_range, PRN_ID, x_GPS, Q)
    % Linear Kalman Filter (propagate once, then update)
    % Inputs:
    %   t: 1 x n_time vector of time epochs                                               (s)
    %   x0: initial state estimate (6x1) [rx; ry; rz; vx; vy; vz]                         (km, km/s)
    %   P0: initial covariance estimate (6x6)                                             
    %   sigma_rho: standard deviation of pseudorange measurements                         (km)
    %   CA_range: n_time x N_sats matrix of pseudorange measurements                      (km)
    %   PRN_ID: n_time x N_sats matrix of PRN IDs (0 if not tracked)
    %   x_GPS: 6 x n_time x N_sats array of GPS satellite states [rx; ry; rz; vx; vy; vz] (km, km/s)
    %   Q: (optional) process noise covariance matrix (6x6). Defaults to zero matrix.    
    % Outputs:
    %   x_est: estimated states over time (6 x n_time)
    %   P_est: estimated covariances over time (6 x 6 x n_time)

    % Initialize arrays to store estimates
    x_est = zeros(6, length(t));
    P_est = zeros(size(P0, 1), size(P0, 2), length(t));

    % Set default process noise matrix if not provided
    if nargin < 8
        Q = zeros(6, 6);
    end

    % Propagate trajectory once using initial state
    x_ref = zeros(6, length(t));
    Phi_ref = cell(length(t), 1);
    x_ref(:, 1) = x0;
    Phi_ref{1} = eye(6);
    
    for k=1:length(t)-1
        [x_ref(:, k+1), Phi_ref{k+1}] = propagation(x_ref(:, k), Phi_ref{k}, t(k+1) - t(k));
    end

    % Update with measurements using propagated trajectory
    P_pred = P0;
    for k=1:length(t)
        % Update with observations at t(k) using propagated state
        tracked_sats = PRN_ID(k, :) > 0;
        z_k = CA_range(k, tracked_sats)'; % Available measurements at time t(k)
        x_t = squeeze(x_GPS(:, k, tracked_sats)); % GPS satellite states for tracked satellites
        
        % Compute measurement matrix H and predicted measurements h_x using propagated state
        [h_x_k, H_k] = observation(x_ref(:, k), x_t);
        
        % Innovation
        dz_k = z_k - h_x_k;
        R_k = (sigma_rho^2) * eye(sum(tracked_sats));
        
        % Kalman Gain
        K_k = (P_pred*H_k') / (H_k*P_pred*H_k' + R_k);
        
        % State update
        x_est(:, k) = x_ref(:, k) + K_k * dz_k;
        
        % Covariance update
        P_est(:, :, k) = (eye(6) - K_k*H_k) * P_pred;

        % Propagate covariance for next measurement (using original trajectory)
        if k < length(t)
            Phi_kk1 = Phi_ref{k} / (Phi_ref{k+1});
            P_pred = Phi_kk1 * P_est(:, :, k) * Phi_kk1.' + Q;
        end
    end
end