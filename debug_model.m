function model = debug_model(num_cams)

import casadi.*

%% dims

nx = 6;
nu = 3;

% cam_pos = [0;0;0.1];
sym_p = SX.sym('pm',3*num_cams,1);
cam_pos = reshape(sym_p,3,num_cams);

nobs = num_cams;       % TODO
halfFOV = deg2rad(45);

%% symbolic variables

x_pw = SX.sym('x',3,1);           % world position
x_vw = SX.sym('v',3,1);          % world velocity
sym_x = vertcat(x_pw, x_vw);
sym_u = SX.sym('u',nu,1);
sym_xdot = SX.sym('xdot',nx,1);     %state derivatives

%% dynamics 
pdot = x_vw;
vdot = -0.1*x_pw + sym_u;

expr_f_expl = vertcat(pdot,vdot);
expr_f_impl = expr_f_expl - sym_xdot;

%% constraints

% take a list of target positions, compute the cosines of their angles with
% the agent's position:
% cos(alpha) = dot(p_target, p_self) / |p_target|*|p_self|

p_rel = cam_pos - x_pw;
sym_obs = sum(([0;0;1].*p_rel)'./sqrt(sum(p_rel.^2,1))',2);
expr_h = sym_obs;
constr_lh = ones(nobs,1)*cos(halfFOV);
constr_uh = ones(nobs,1)*1e1;

func_h = Function('obs',{x_pw,sym_p},{sym_obs});      % TODO

%% measurements (nonlinear_ls)

expr_y = [sym_x; sym_obs; sym_u];
expr_y_e = [sym_x; sym_obs];

%% external cost
yr_u = zeros(nu, 1);
yr_obs = ones(nobs, 1);
yr_x = zeros(nx, 1);

dWx = 1e0*ones(6,1);
dWobs = 1e1*ones(nobs, 1);
dWu = 1e-2*ones(nu, 1);

ymyr = [sym_x; sym_obs; sym_u] - [yr_x; yr_obs; yr_u];

expr_ext_cost = 0.5 * ymyr' * ([dWx; dWobs; dWu] .* ymyr);
W = diag([dWx; dWobs; dWu]);
W_e = diag([dWx; dWobs]);

%% populate structure
model.nx = nx;                      % OK
model.nu = nu;                      % OK
model.nobs = nobs;
model.np = 3*nobs;     % TODO: disambiguate nobs, np/num_targets

model.ny = 6 + nu + nobs;
model.ny_e = 6 + nobs;
model.sym_x = sym_x;                % OK
model.sym_xdot = sym_xdot;          % OK
model.sym_u = sym_u;                % OK
model.expr_f_expl = expr_f_expl;    % OK
model.expr_f_impl = expr_f_impl;    % OK
model.sym_p = sym_p;              % TODO

model.expr_h = expr_h;              % nonlinear path (x,u) inequality constraint
model.constr_lh = constr_lh;
model.constr_uh = constr_uh;
model.func_h = func_h;
model.expr_y = expr_y;              % nonlinear y(x,u) for NL-LS cost function ( |y - yr|^2 )
model.expr_y_e = expr_y_e;              % nonlinear y(x,u) for NL-LS cost function ( |y - yr|^2 )
model.expr_ext_cost = expr_ext_cost;    % custom stage cost function
model.expr_ext_cost_e = expr_ext_cost;    % custom stage cost function
model.W = W;
model.W_e = W_e;
end