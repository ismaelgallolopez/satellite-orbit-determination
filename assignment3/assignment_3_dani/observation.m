
function [h_x, H] = observation(x_r, x_t)
    % Observation model for pseudorange measurements
    % Inputs:
    %   x_r: 6x1 vector [rx; ry; rz; vx; vy; vz] of receiver
    %   x_t: 6xN_sats matrix [rx; ry; rz; vx; vy; vz] for each satellite
    % Outputs:
    %   h_x: N_sats x 1 vector of predicted pseudorange measurements
    %   H: N_sats x 6 design matrix
    N_sats = size(x_t, 2);
    h_x = zeros(N_sats, 1);
    H = zeros(N_sats, 6);
    
    r_t = x_t(1:3, :);
    r_r = x_r(1:3);

    for i = 1:N_sats
        dist = norm(r_t(:, i) - r_r);
        h_x(i) = dist;
        % Design matrix
        H(i, 1:3) = (r_r - r_t(:, i))' / dist;
    end
end