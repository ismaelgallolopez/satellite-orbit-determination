%% 
clc; clear; close;

% Load all data files from the input directory, the variables will be named after the file names
load_directory_data('input');
load_directory_data('ref', '_ref');
if ~exist('output', 'dir')
    mkdir('output');
end

% Constants
C = constants();

% Format input data
x_GPS = permute(cat(3, rx_gps, ry_gps, rz_gps, vx_gps, vy_gps, vz_gps), [3, 1, 2]); % 6 x n_time x N_sats
x_ref = [rx_ref'; ry_ref'; rz_ref'; vx_ref'; vy_ref'; vz_ref']; % 6 x n_time

% Q5. Initial propagation
x1 = [rx_ref(1); ry_ref(1); rz_ref(1); vx_ref(1); vy_ref(1); vz_ref(1)];
[x2, Phi2] = propagation(x1, eye(6), t(2)-t(1)); % Propagate to t2

fprintf('Q5:\n');
fprintf('   Propagated state at t2:\n');
fprintf('   r(t2) = %.6f %.6f %.6f km \n', x2(1:3));
fprintf('   v(t2) = %.6f %.6f %.6f km/s \n', x2(4:6));
fprintf('   STM at t2:\n');
fprintf('   Phi_22(t2, t1) = %.10f \n', Phi2(2,2));
fprintf('   Phi_15(t2, t1) = %.10f \n', Phi2(1,5));
fprintf('   Phi_36(t2, t1) = %.10f \n', Phi2(3,6));
fprintf('   Phi_52(t2, t1) = %.10f \n', Phi2(5,2));

% EKF without process noise
sigma_rho = 3e-3;   % km
sigma_pos = 2e-3;   % km
sigma_vel = 0.1e-3; % km/s
rho_r_v = 0.7;      % correlation coefficient between position and velocity

% Construct P0 (6x6 covariance matrix)
P0 = diag([sigma_pos^2, sigma_pos^2, sigma_pos^2, ...
           sigma_vel^2, sigma_vel^2, sigma_vel^2]);
P0(1,4) = rho_r_v * sigma_pos * sigma_vel;
P0(4,1) = P0(1,4);
P0(2,5) = rho_r_v * sigma_pos * sigma_vel;
P0(5,2) = P0(2,5);
P0(3,6) = rho_r_v * sigma_pos * sigma_vel;
P0(6,3) = P0(3,6);

% Run EKF without process noise
[x_est_ekf_nn, P_est_ekf_nn, ~] = ekf(t, x1, P0, sigma_rho, CA_range, PRN_ID, x_GPS); % No process noise
d_r_ekf_nn = vecnorm(x_est_ekf_nn(1:3, :) - x_ref(1:3, :), 2, 1); % Position error over time no process noise
d_v_ekf_nn = vecnorm(x_est_ekf_nn(4:6, :) - x_ref(4:6, :), 2, 1); % Velocity error over time no process noise
sigma_r_ekf_nn = sqrt(squeeze(P_est_ekf_nn(1,1,:) + P_est_ekf_nn(2,2,:) + P_est_ekf_nn(3,3,:))); % Position 3-sigma no process noise
sigma_v_ekf_nn = sqrt(squeeze(P_est_ekf_nn(4,4,:) + P_est_ekf_nn(5,5,:) + P_est_ekf_nn(6,6,:))); % Velocity 3-sigma no process noise

fprintf('\nQ6:\n');
fprintf('   EKF without process noise:\n');

% Position table
fprintf('   Epoch 10: r = (%.6f, %.6f, %.6f) km\n', x_est_ekf_nn(1:3, 10));
fprintf('   Epoch 20: r = (%.6f, %.6f, %.6f) km\n', x_est_ekf_nn(1:3, 20));
fprintf('   Epoch 30: r = (%.6f, %.6f, %.6f) km\n', x_est_ekf_nn(1:3, 30));

% Velocity table
fprintf('   Epoch 10: v = (%.6f, %.6f, %.6f) km/s\n', x_est_ekf_nn(4:6, 10));
fprintf('   Epoch 20: v = (%.6f, %.6f, %.6f) km/s\n', x_est_ekf_nn(4:6, 20));
fprintf('   Epoch 30: v = (%.6f, %.6f, %.6f) km/s\n', x_est_ekf_nn(4:6, 30));

% Run EKF with process noise
sigma_a = 0.01e-3; % km
sigma_b = 0.01e-3; % km/s
Q = [(sigma_a^2)*eye(3), zeros(3); zeros(3), (sigma_b^2)*eye(3)]; % Process noise covariance matrix
[x_est_ekf_wn, P_est_ekf_wn, obs_res_wn] = ekf(t, x1, P0, sigma_rho, CA_range, PRN_ID, x_GPS, Q); % With process noise
d_r_ekf_wn = vecnorm(x_est_ekf_wn(1:3, :) - x_ref(1:3, :), 2, 1); % Position error over time with process noise
d_v_ekf_wn = vecnorm(x_est_ekf_wn(4:6, :) - x_ref(4:6, :), 2, 1); % Velocity error over time with process noise
sigma_r_ekf_wn = sqrt(squeeze(P_est_ekf_wn(1,1,:) + P_est_ekf_wn(2,2,:) + P_est_ekf_wn(3,3,:))); % Position 3-sigma with process noise
sigma_v_ekf_wn = sqrt(squeeze(P_est_ekf_wn(4,4,:) + P_est_ekf_wn(5,5,:) + P_est_ekf_wn(6,6,:))); % Velocity 3-sigma with process noise

%% Plotting
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultAxesFontSize', 16);
set(groot, 'defaultLegendFontSize', 16);
set(groot, 'defaultAxesLabelFontSizeMultiplier', 20/16);
set(groot, 'defaultAxesTitleFontSizeMultiplier', 22/16);
set(groot, 'defaultAxesLineWidth', 1.2);
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesGridAlpha', 0.3);
set(groot, 'defaultAxesMinorGridAlpha', 0.15);
set(groot, 'DefaultFigureVisible', 'off');

% Q7
fig = figure('Position',[200 200 1600 900]);
yyaxis left
h1 = plot(t-t(1), d_r_ekf_nn*1000);
ylabel('Position Error [m]');
yyaxis right
h2 = plot(t-t(1), d_v_ekf_nn*1000);
ylabel('Velocity Error [m/s]');
xlabel('Elapsed Time [s]');
title('EKF Position and Velocity Error without Process Noise');
grid on;
saveas(fig, fullfile('output','error_ekf_nn.png'));
close(fig);

% Q8
fig = figure('Position',[200 200 1600 900]);
yyaxis left
h1 = plot(t-t(1), sigma_r_ekf_nn*1000);
set(gca, 'YScale', 'log');
ylabel('Position $\sigma$ [m]');
yyaxis right
h2 = plot(t-t(1), sigma_v_ekf_nn*1000);
set(gca, 'YScale', 'log');
ylabel('Velocity $\sigma$ [m/s]');
xlabel('Elapsed Time [s]');
title('EKF Position and Velocity Standard Deviation without Process Noise');
grid on;
saveas(fig, fullfile('output','sigma_ekf_nn.png'));
close(fig);

% Q14
fig = figure('Position',[200 200 1600 1000]);
subplot(2,1,1);
yyaxis left
h1 = plot(t-t(1), d_r_ekf_wn*1000);
hold on;
h2 = plot(t-t(1), d_r_ekf_nn*1000, '--', Color = h1.Color);
ylabel('Position Error [m]');
yyaxis right
h3 = plot(t-t(1), d_v_ekf_wn*1000);
hold on;
h4 = plot(t-t(1), d_v_ekf_nn*1000, '--', Color = h3.Color);
ylabel('Velocity Error [m/s]');
xlabel('Elapsed Time [s]');
% Empty plots to show legend in black
e1 = plot(nan, nan, '-k', 'DisplayName', 'No Process Noise');
e2 = plot(nan, nan, '--k', 'DisplayName', 'With Process Noise');
legend([e1, e2], 'Location', 'best');
title('EKF Position and Velocity Error');
grid on;
subplot(2,1,2);
yyaxis left
h5 = plot(t-t(1), sigma_r_ekf_wn*1000);
hold on;
h6 = plot(t-t(1), sigma_r_ekf_nn*1000, '--', Color = h5.Color);
set(gca, 'YScale', 'log');
ylabel('Position $\sigma$ [m]');
yyaxis right
h7 = plot(t-t(1), sigma_v_ekf_wn*1000);
hold on;
h8 = plot(t-t(1), sigma_v_ekf_nn*1000, '--', Color = h7.Color);
set(gca, 'YScale', 'log');
ylabel('Velocity $\sigma$ [m/s]');
xlabel('Elapsed Time [s]');
title('EKF Position and Velocity Standard Deviation');
grid on;
saveas(fig, fullfile('output','ekf_wn.png'));
close(fig);

% Q16
fig = figure('Position',[200 200 1600 900]);
h1 = plot(t-t(1), d_r_ekf_wn*1000, 'DisplayName', 'Position Error $\delta r$');
hold on;
h2 = plot(t-t(1), obs_res_wn*1000, 'DisplayName', 'Observation Residual RMS $e_{RMS}$');
ylabel('Error [m]');
xlabel('Elapsed Time [s]');
title('EKF Position Error and Observation Residual RMS with Process Noise');
legend('Location', 'best');
grid on;
saveas(fig, fullfile('output','obs_res.png'));
close(fig);

%% EXTRA: LKF
[x_est_lkf, P_est_lkf] = lkf(t, x1, P0, sigma_rho, CA_range, PRN_ID, x_GPS); % No process noise
d_r_lkf = vecnorm(x_est_lkf(1:3, :) - x_ref(1:3, :), 2, 1); % Position error over time no process noise
d_v_lkf = vecnorm(x_est_lkf(4:6, :) - x_ref(4:6, :), 2, 1); % Velocity error over time no process noise
sigma_r_lkf = sqrt(squeeze(P_est_lkf(1,1,:) + P_est_lkf(2,2,:) + P_est_lkf(3,3,:))); % Position 3-sigma no process noise
sigma_v_lkf = sqrt(squeeze(P_est_lkf(4,4,:) + P_est_lkf(5,5,:) + P_est_lkf(6,6,:))); % Velocity 3-sigma no process noise

fig = figure('Position',[200 200 1600 900]);
yyaxis left
h1 = plot(t-t(1), d_r_lkf*1000);
hold on;
h2 = plot(t-t(1), d_r_ekf_nn*1000, '--', Color = h1.Color);
ylabel('Position Error [m]');
yyaxis right
h3 = plot(t-t(1), d_v_lkf*1000);
hold on;
h4 = plot(t-t(1), d_v_ekf_nn*1000, '--', Color = h3.Color);
ylabel('Velocity Error [m/s]');
xlabel('Elapsed Time [s]');
title('LKF vs EKF Position and Velocity Error without Process Noise');
% Empty plots to show legend in black
e1 = plot(nan, nan, '-k', 'DisplayName', 'LKF');
e2 = plot(nan, nan, '--k', 'DisplayName', 'EKF');
legend([e1, e2], 'Location', 'best');
grid on;
saveas(fig, fullfile('output','error_lkf.png'));
close(fig);

fig = figure('Position',[200 200 1600 900]);
yyaxis left
h1 = plot(t-t(1), sigma_r_lkf*1000);
hold on;
h2 = plot(t-t(1), sigma_r_ekf_nn*1000, '--', Color = h1.Color);
ylabel('Position $\sigma$ [m]');
yyaxis right
h3 = plot(t-t(1), sigma_v_lkf*1000);
hold on;
h4 = plot(t-t(1), sigma_v_ekf_nn*1000, '--', Color = h3.Color);
ylabel('Velocity $\sigma$ [m/s]');
xlabel('Elapsed Time [s]');
title('LKF vs EKF Position and Velocity Standard Deviation without Process Noise');
% Empty plots to show legend in black
e1 = plot(nan, nan, '-k', 'DisplayName', 'LKF');
e2 = plot(nan, nan, '--k', 'DisplayName', 'EKF');
legend([e1, e2],'Location', 'best');
grid on;
saveas(fig, fullfile('output','sigma_lkf.png'));
close(fig);