function C = constants
    % Define constants only once (persistent)
    persistent CONST
    if isempty(CONST)
        % Spacecraft and environmental parameters
        CONST.mass_SC =     600; % kg, Spacecraft Mass
        CONST.Area =        1; % m^2, Reference Surface Area
        CONST.C_D =         2.6; % Aerodynamic Drag Coefficient
        CONST.C_20 = 	    -4.841692151273e-04; % Gravity Field Coefficient
        CONST.rho_atm =     1e-11; % kg/m^3, Atmospheric Density
        
        % Physical constants
        CONST.c_light =     299792.458; % km/s, Speed of Light
        CONST.omega_Earth = 7.292115e-05; % rad/s, Earth Rotation Rate
        CONST.GM_Earth =    3.986004415e05; % km^3/s^2, Earth's Gravitational Constant
        CONST.R_Earth =     6378.1366; % km, Earth's Reference Radius
    end
    C = CONST;
end
