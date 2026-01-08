function [x_est, P_est, obs_res] = ekf(t, x0, P0, sigma_rho, CA_range, PRN_ID, x_GPS, Q)
    % Extended Kalman Filter
    % Inputs:
    %   t: 1 x n_time vector of time epochs                                               (s)
    %   x0: initial state estimate (4x1) [rx; ry; rz; vx; vy; vz]                         (km, km/s)
    %   P0: initial covariance estimate (6x6)                                             
    %   sigma_rho: standard deviation of pseudorange measurements                         (km)
    %   CA_range: n_time x N_sats matrix of pseudorange measurements                      (km)
    %   PRN_ID: n_time x N_sats matrix of PRN IDs (0 if not tracked)
    %   x_GPS: 6 x n_time x N_sats array of GPS satellite states [rx; ry; rz; vx; vy; vz] (km, km/s)
    %   Q: (optional) process noise covariance matrix (6x6). Defaults to zero matrix.    
    % Outputs:
    %   x_est: estimated states over time (6 x n_time)
    %   P_est: estimated covariances over time (6 x 6 x n_time)
    %   obs_res: RMS per epoch of the observation residuals over time

    % Initialize arrays to store estimates
    x_est = zeros(6, length(t));
    P_est = zeros(size(P0, 1), size(P0, 2), length(t));
    obs_res = zeros(length(t), 1);

    % Set default process noise matrix if not provided
    if nargin < 8
        Q = zeros(6, 6);
    end

    x_pred = x0;
    P_pred = P0;
    for k=1:length(t)
        % Update with observations at t(k)
        tracked_sats = PRN_ID(k, :) > 0;
        z_k = CA_range(k, tracked_sats)'; % Available measurements up to time t(k)
        x_t = squeeze(x_GPS(:, k, tracked_sats)); % GPS satellite states for tracked satellites
        [h_x_k, H_k] = observation(x_pred, x_t);
        dz_k = z_k - h_x_k;
        R_k = (sigma_rho^2) * eye(sum(tracked_sats));
        % Kalman Gain
        K_k = (P_pred*H_k') / (H_k*P_pred*H_k' + R_k);
        x_est(:, k) = x_pred + K_k * dz_k;
        P_est(:, :, k) = (eye(6) - K_k*H_k) * P_pred;

        % Compute observation residual RMS
        dx_k = K_k * dz_k;
        e_k = dz_k - H_k * dx_k;
        obs_res(k) = sqrt(mean(e_k.^2));

        % Propagation
        if k==length(t)
            % Last epoch, no need to propagate
            break;
        end
        dt = t(k+1) - t(k);
        [x_pred, Phi] = propagation(x_est(:, k), eye(6), dt);
        P_pred = Phi * P_est(:, :, k) * Phi.' + Q;
    end
end