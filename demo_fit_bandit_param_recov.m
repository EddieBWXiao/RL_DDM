
% Bowen Xiao 20250308
% fitting the bandit task with RLDDM + demonstrate param recov 
% requires an adapted mfit package in an adjacent folder

addpath('../mfit')
% note: folder should have mvis_scat_ref_corr

%% fit
data = load_bandit_data;
results = fit_bandit(data);

%% create task for simulation
task.C = 2;
task.N = 400;

% generate a simple 80-20 PRL:
task.r = [binornd(1,0.8,[task.N/4,1]), binornd(1,0.2,[task.N/4,1]);
    binornd(1,0.2,[task.N/4,1]), binornd(1,0.8,[task.N/2,1]);
    binornd(1,0.8,[task.N/4,1]), binornd(1,0.2,[task.N/4,1]);
    binornd(1,0.2,[task.N/4,1]), binornd(1,0.8,[task.N/2,1])];
task.s = ones(task.N,1); %one state

%% parameter recovery: run
for i = 1:size(results.x,1)
    sim_data(i) = simfun_bandit(results.x(i,:),task);
end
recov = fit_bandit(sim_data);

%% parameter recovery: visualise
subplot(2,2,1)
mvis_scat_ref_corr(results.x(:,1),recov.x(:,1))
xlabel(sprintf('Simulated %s',results.param(1).name))
ylabel(sprintf('Recovered %s',results.param(1).name))
subplot(2,2,2)
mvis_scat_ref_corr(results.x(:,2),recov.x(:,2))
xlabel(sprintf('Simulated %s',results.param(2).name))
ylabel(sprintf('Recovered %s',results.param(2).name))
subplot(2,2,3)
mvis_scat_ref_corr(results.x(:,3),recov.x(:,3))
xlabel(sprintf('Simulated %s',results.param(3).name))
ylabel(sprintf('Recovered %s',results.param(3).name))
subplot(2,2,4)
mvis_scat_ref_corr(results.x(:,4),recov.x(:,4))
xlabel(sprintf('Simulated %s',results.param(4).name))
ylabel(sprintf('Recovered %s',results.param(4).name))

