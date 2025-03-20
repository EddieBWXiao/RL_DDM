% Bowen Xiao 20250319
% fitting the sigmoid-hat DDM and demonstrate parameter recovery
% requires an adapted mfit package in an adjacent folder

addpath(genpath('../mfit'))
% note: folder should have mvis_scat_ref_corr

rng(22222)

%% create task for simulation
task.C = 2;
task.N = 200;

% generate stimulus values ranging from -5 to 5
stim_values = linspace(-5, 5, 11);
task.s = stim_values(randi(length(stim_values), [task.N, 1]));

%% generate parameters from priors for simulation
n_sim_subjects = 40; % number of simulated subjects

% parameter ranges
kappa_lb = 1e-3; kappa_ub = 10;
a_lb = 1e-3; a_ub = 4;
w_alpha = 1.1; w_beta = 1.1; % beta distribution parameters
T_lb = 0; T_ub = 1;

% sample from priors
sim_params = zeros(n_sim_subjects, 4);
sim_params(:,1) = unifrnd(kappa_lb, kappa_ub, n_sim_subjects, 1); % kappa
sim_params(:,2) = unifrnd(a_lb, a_ub, n_sim_subjects, 1); % a
%sim_params(:,3) = betarnd(w_alpha, w_beta, n_sim_subjects, 1); % w (from beta distribution)
%sim_params(:,3) = 0.5*ones(n_sim_subjects,1);
sim_params(:,3) = unifrnd(0.3, 0.7, n_sim_subjects, 1);
sim_params(:,4) = unifrnd(T_lb, T_ub, n_sim_subjects, 1); % T


%% parameter recovery: run
%sim_data = [];
recov = [];
for i = 1:n_sim_subjects
    sim_data(i) = simfun_ddm_sgmhat(sim_params(i,:), task);
end
recov = fit_ddm_sgmhat(sim_data);

%% parameter recovery: visualise
subplot(2,2,1)
mvis_scat_ref_corr(sim_params(:,1),recov.x(:,1))
xlabel(sprintf('Simulated %s', 'kappa'))
ylabel(sprintf('Recovered %s', 'kappa'))
subplot(2,2,2)
mvis_scat_ref_corr(sim_params(:,2),recov.x(:,2))
xlabel(sprintf('Simulated %s','a'))
ylabel(sprintf('Recovered %s','a'))
subplot(2,2,3)
mvis_scat_ref_corr(sim_params(:,3),recov.x(:,3))
xlabel(sprintf('Simulated %s','w'))
ylabel(sprintf('Recovered %s','w'))
subplot(2,2,4)
mvis_scat_ref_corr(sim_params(:,4),recov.x(:,4))
xlabel(sprintf('Simulated %s','T'))
ylabel(sprintf('Recovered %s','T'))