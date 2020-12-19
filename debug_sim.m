%
% Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
% Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
% Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
% Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
%
% This file is part of acados.
%
% The 2-Clause BSD License
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.;
%

%% test of native matlab interface
clear all


% check that env.sh has been run
env_run = getenv('ENV_RUN');
if (~strcmp(env_run, 'true'))
	error('env.sh has not been sourced! Before executing this example, run: source env.sh');
end


%% arguments
compile_interface = 'auto';
codgen_model = 'true';
% simulation
sim_sens_forw = 'false';
sim_num_stages = 4;
sim_num_steps = 1;
%shooting_nodes = [0 0.1 0.2 0.3 0.5 1];
ocp_N = 30;
model_name = 'quad_notilt';

nlp_solver = 'sqp';
% nlp_solver = 'sqp_rti';   % real-time solver, only does 1 iter per loop
nlp_solver_exact_hessian = 'false';
% nlp_solver_exact_hessian = 'true';
regularize_method = 'no_regularize';
%regularize_method = 'project';
%regularize_method = 'mirror';
% regularize_method = 'convexify';        % supposed to be the best
nlp_solver_max_iter = 100;
nlp_solver_ext_qp_res = 1;
% qp_solver = 'partial_condensing_osqp';
% qp_solver = 'partial_condensing_qpoases';     % try sqp with warm-start
qp_solver = 'partial_condensing_hpipm';
% qp_solver = 'full_condensing_hpipm';
qp_solver_cond_N = 5;       % adjustment param for partial condensing
dyn_type = 'explicit';
% dyn_type = 'implicit';
ocp_sim_method = 'erk';
% ocp_sim_method = 'irk';
% ocp_sim_method = 'irk_gnsf';
ocp_sim_method_num_stages = 4;
ocp_sim_method_num_steps = 3;
%cost_type = 'linear_ls';
cost_type = 'nonlinear_ls';
% cost_type = 'auto';

%% create model entries

model = debug_model(1);       % TODO

% dims
T = 6.0; % horizon length time
nx = model.nx;
nu = model.nu;
nobs = model.nobs;
ny = model.ny;
ny_e = model.ny_e;
np = model.np;        % TODO

% constraint formulation
% external function constraint
nbx = 0;
nbu = 0;
ng = 0;
nh = nu+nobs;
nh_e = 0;
nsh = nobs;

% constraints
x0 = zeros(nx, 1);
lh = model.constr_lh;
uh = model.constr_uh;

%% acados ocp model
ocp_model = acados_ocp_model();
ocp_model.set('name', model_name);
ocp_model.set('T', T);
% dims
ocp_model.set('dim_nx', nx);
ocp_model.set('dim_nu', nu);
ocp_model.set('dim_ny', ny);
ocp_model.set('dim_ny_e', ny_e);
ocp_model.set('dim_nh', nh);
ocp_model.set('dim_np', np);

% symbolics
ocp_model.set('sym_x', model.sym_x);
ocp_model.set('sym_u', model.sym_u);
ocp_model.set('sym_xdot', model.sym_xdot);
ocp_model.set('sym_p', model.sym_p);

% cost
ocp_model.set('cost_type', 'nonlinear_ls');
ocp_model.set('cost_type_e', 'nonlinear_ls');
ocp_model.set('cost_expr_y', model.expr_y);
ocp_model.set('cost_W', model.W);
ocp_model.set('cost_expr_y_e', model.expr_y_e);
ocp_model.set('cost_W_e', model.W_e);

% dynamics
ocp_model.set('dyn_type', 'explicit');
ocp_model.set('dyn_expr_f', model.expr_f_expl);

% constraints
ocp_model.set('constr_x0', x0);
ocp_model.set('constr_expr_h', model.expr_h);
ocp_model.set('constr_lh', lh);
ocp_model.set('constr_uh', uh);

%% acados ocp opts
ocp_opts = acados_ocp_opts();
ocp_opts.set('compile_interface', compile_interface);
ocp_opts.set('codgen_model', codgen_model);
ocp_opts.set('param_scheme_N', ocp_N);
ocp_opts.set('nlp_solver', nlp_solver);
ocp_opts.set('nlp_solver_exact_hessian', nlp_solver_exact_hessian);
ocp_opts.set('regularize_method', regularize_method);
ocp_opts.set('nlp_solver_ext_qp_res', nlp_solver_ext_qp_res);
ocp_opts.set('nlp_solver_max_iter', nlp_solver_max_iter);
ocp_opts.set('qp_solver', qp_solver);
ocp_opts.set('qp_solver_cond_N', qp_solver_cond_N);
ocp_opts.set('sim_method', ocp_sim_method);
ocp_opts.set('sim_method_num_stages', ocp_sim_method_num_stages);
ocp_opts.set('sim_method_num_steps', ocp_sim_method_num_steps);

    
%% acados ocp
% create ocp
ocp = acados_ocp(ocp_model, ocp_opts);


%% acados sim model
sim_model = acados_sim_model();
% dims
sim_model.set('dim_nx', nx);
sim_model.set('dim_nu', nu);
sim_model.set('dim_np', np);  % TODO
% symbolics
sim_model.set('sym_x', model.sym_x);
sim_model.set('sym_u', model.sym_u);
sim_model.set('sym_xdot', model.sym_xdot);
sim_model.set('sym_p', model.sym_p);
% model
sim_model.set('T', T/ocp_N);
sim_model.set('dyn_type', 'explicit');
sim_model.set('dyn_expr_f', model.expr_f_expl);


%% acados sim opts
sim_opts = acados_sim_opts();
sim_opts.set('compile_interface', compile_interface);
sim_opts.set('codgen_model', codgen_model);
sim_opts.set('num_stages', sim_num_stages);
sim_opts.set('num_steps', sim_num_steps);
sim_opts.set('method', ocp_sim_method);
sim_opts.set('sens_forw', sim_sens_forw);


%% acados sim
% create sim
sim = acados_sim(sim_model, sim_opts);


%% closed loop simulation
n_sim = 200;
x_sim = zeros(nx, n_sim+1);
x_sim(:,1) = x0; % initial state
u_sim = zeros(nu, n_sim);
s_sim = zeros(nobs, n_sim);

sqp_iter_sim = zeros(n_sim,1);

% set trajectory initialization
x_traj_init = repmat(x0, 1, ocp_N+1);
u_traj_init = zeros(nu, ocp_N); % repmat(u0_ref, 1, ocp_N);

% precompute references
t_ref = (0:n_sim-1+ocp_N)*(T/ocp_N); n_ref = size(t_ref,2);
y_ref = [3.0*cos(0.3*t_ref); 3.0*sin(0.3*t_ref); -2.0*ones(1,n_ref);    % positions
        zeros(3,n_ref); 
        ones(model.nobs,n_ref);
        zeros(nu,n_ref)];
target_ref = repmat(reshape([0;0;0.1],[],1),1,n_ref);
p_ref = reshape(target_ref,[],n_ref);

for ii=1:n_sim

   fprintf('\nsimulation step %d\n', ii);

    tic

    % set x0
    ocp.set('constr_x0', x_sim(:,ii));

    % set reference (different at each stage)
    for jj=0:ocp_N-1
        ocp.set('cost_y_ref', y_ref(:,ii+jj), jj);
        ocp.set('p', p_ref(:,ii+jj)', jj);     % TODO
    end
    ocp.set('cost_y_ref_e', y_ref(1:ny_e,ii+ocp_N));

    % set trajectory initialization (if not, set internally using previous solution)
    ocp.set('init_x', x_traj_init);
    ocp.set('init_u', u_traj_init);

    % solve
    ocp.solve();

    % get solution
    x = ocp.get('x');
    u = ocp.get('u');

    % store first input
    u_sim(:,ii) = ocp.get('u', 0);

    % set initial state of sim
    sim.set('x', x_sim(:,ii));
    % set input in sim
    sim.set('u', u_sim(:,ii));
    % set parameter value
    sim.set('p', p_ref(:,ii));

    % simulate state
    sim.solve();

    % get new state
    x_sim(:,ii+1) = sim.get('xn');
    x_traj_init = [x(:,2:ocp_N+1), x(:,ocp_N+1)];
    u_traj_init = [u(:,2:ocp_N), u(:,ocp_N)];

    time_ext = toc;

    status = ocp.get('status');
    sqp_iter = ocp.get('sqp_iter');
    time_tot = ocp.get('time_tot');
    time_lin = ocp.get('time_lin');
    time_reg = ocp.get('time_reg');
    time_qp_sol = ocp.get('time_qp_sol');

    sqp_iter_sim(ii) = sqp_iter;
    if status ~= 0
        warning('ocp_nlp solver returned nonzero status!');
    end

    fprintf('\nstatus = %d, sqp_iter = %d, time_ext = %f [ms], time_int = %f [ms] (time_lin = %f [ms], time_qp_sol = %f [ms], time_reg = %f [ms])\n', status, sqp_iter, time_ext*1e3, time_tot*1e3, time_lin*1e3, time_qp_sol*1e3, time_reg*1e3);
    
    if 0
        % print statistics
        ocp.print('stat')
    end
    if status==0
        fprintf('\nsuccess!\n\n');
    else
        fprintf('\nsolution failed!\n\n');
    end
end


%% plot

time_sim = t_ref(1:n_sim+1);
ref_hist = [y_ref, y_ref(:,end)]; ref_hist=ref_hist(1:ny,:);
f=figure();
plot3(x_sim(1,:), x_sim(2,:), x_sim(3,:),'b'); axis equal; hold on
f.CurrentAxes.ZDir = 'Reverse'; grid on;
% plot3(  x(cbHistory<cos(param.halfFOV),1), ...
%         x(cbHistory<cos(param.halfFOV),2), ...
%         x(cbHistory<cos(param.halfFOV),3),'p','LineWidth',2);
plot3(ref_hist(1,:), ref_hist(2,:), ref_hist(3,:),'r--');
xlabel("X"); ylabel("Y"); zlabel("Z");

figure()
subplot(321)
plot(time_sim, x_sim(1,:)','b'); ylabel("X");
hold on; plot(time_sim, ref_hist(1,1:n_sim+1)','r--');
subplot(323)
plot(time_sim, x_sim(2,:)','b'); ylabel("Y");
hold on; plot(time_sim, ref_hist(2,1:n_sim+1)','r--');
subplot(325)
plot(time_sim, x_sim(3,:)','b'); ylabel("Z");
hold on; plot(time_sim, ref_hist(3,1:n_sim+1)','r--');
subplot(322)
plot(time_sim, (x_sim(4,:)'),'b'); ylabel("\phi");
subplot(324)
plot(time_sim, (x_sim(5,:)'),'b'); ylabel("\theta");
subplot(326)
plot(time_sim, (x_sim(6,:))','b'); ylabel("\psi");

figure()
for i=1:3
stairs(time_sim(1:end-1), u_sim(i,:)'); ylabel(join(["u_",i])); hold on
end
hold off

%%
outp = zeros(nobs, n_sim+1);
for ii=1:n_sim+1
tmp=model.func_h(x_sim(1:3,ii), p_ref(:,ii));    % TODO
outp(:,ii)=tmp.full();
end
figure()
subplot(211)
plot(time_sim,outp);
subplot(212)
plot(time_sim(1:end-1),s_sim);
