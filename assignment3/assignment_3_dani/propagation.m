function [x, Phi] = propagation(x0, Phi0, dt)
% Propagate 6x1 state and 6x6 STM over a time step dt (s) using two-body dynamics
% Inputs:
%   x0: initial 6x1 state [r; v] in km and km/s (ECEF/ECR frame).
%   Phi0: initial 6x6 state transition matrix.
%   dt: propagation interval in seconds (can be negative).
% Outputs:
%   x: propagated state at t0+dt.
%   Phi: propagated STM (6x6), initialized as identity at t0.

	% Augmented initial state: state + vec(I6)
	g0 = [x0(:); Phi0(:)];

	% ODE integration with ode45
	odefun = @(t, g) dgn(t, g);
	tspan = [0, dt];
	opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
	[~, g] = ode45(odefun, tspan, g0, opts);

	g_end = g(end, :).';
	x = g_end(1:6);
	Phi = reshape(g_end(7:end), 6, 6);
end

function gdot = dgn(~, g)
	r = g(1:3);
	v = g(4:6);
	Phi = reshape(g(7:end), 6, 6);

    C = constants();
	mu = C.GM_Earth; % km^3/s^2

	% Earth rotation skew matrix
	Omega = [ 0             -C.omega_Earth 0;
			  C.omega_Earth 0              0;
			  0             0              0]; % rad/s

	r_norm = norm(r);
	% Two-body + Earth rotation accelerations
	a_grav = -(mu / r_norm^3) * r;
	a_rot = -(Omega * Omega) * r - 2 * Omega * v;
	a = a_grav + a_rot;

	% State Jacobian F
	F = [zeros(3), eye(3);
		 (mu / r_norm^5) * (3 * (r * r.') - (r_norm^2) * eye(3)) - (Omega * Omega), -2 * Omega];

	Phi_dot = F * Phi;

	gdot = [v; a; Phi_dot(:)];
end