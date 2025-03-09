
% Bowen Xiao 20250308
% fitting the bandit task with RLDDM
% demonstrate model validation (generative performance / ppc)
% requires an adapted mfit package in an adjacent folder

addpath(genpath('../mfit'))
% note: folder should have mvis_scat_ref_corr

%% fit
data = load_bandit_data;
results = fit_bandit(data);

%% simulate for each participant
for i = 1:size(results.x,1)
    
    task.C = data(i).C;
    task.N = data(i).N;
    task.s = data(i).s; % copy the state
    % reconstruct the task!
    task.r = nan(task.N, task.C);
    for iTrial = 1:task.N
        % for the chosen option
        task.r(iTrial,data(i).c(iTrial)) = data(i).r(iTrial);
        % for the unchosen (2x-1, flipping 1 & 2 code)
        task.r(iTrial, 2 - data(i).c(iTrial) + 1) = 1 - data(i).r(iTrial);
    end
    
    sim_data(i) = simfun_bandit(results.x(i,:),task);
end

%% examine reaction time distribution
% Extract all RT values and choice data from both datasets
real_rt = [];
sim_rt = [];
real_choice = [];
sim_choice = [];
ptp_id = [];

% Concatenate all RT values and choice data from the data struct
for i = 1:length(data)
    if ~isempty(data(i).rt) && ~isempty(data(i).c)
        n_trials = length(data(i).rt);
        real_rt = [real_rt; data(i).rt(:)];
        real_choice = [real_choice; data(i).c(:)];
        % Create participant ID vector (same length as rt for each participant)
        ptp_id = [ptp_id; repmat(i, n_trials, 1)];
    end
end

% Concatenate all RT values and choice data from the sim_data struct
sim_ptp_id = [];
for i = 1:length(sim_data)
    if ~isempty(sim_data(i).rt) && ~isempty(sim_data(i).c)
        n_trials = length(sim_data(i).rt);
        sim_rt = [sim_rt; sim_data(i).rt(:)];
        sim_choice = [sim_choice; sim_data(i).c(:)];
        % Create participant ID vector for simulated data
        sim_ptp_id = [sim_ptp_id; repmat(i, n_trials, 1)];
    end
end

% Plot overall density comparison
figure;
hold on;

% Plot the real RT density
[f_real, xi_real] = ksdensity(real_rt);
plot(xi_real, f_real, 'b-', 'LineWidth', 2);

% Plot the simulated RT density
[f_sim, xi_sim] = ksdensity(sim_rt);
plot(xi_sim, f_sim, 'r--', 'LineWidth', 2);

% Add labels and legend
xlabel('Response Time (RT)');
ylabel('Density');
title('Comparison of Real vs. Simulated Response Time Distributions');
legend('Real RT', 'Simulated RT');

% Plot density by choice
figure;

% Separate real data by choice
real_choice1_rt = real_rt(real_choice == 1);
real_choice2_rt = real_rt(real_choice == 2);

% Separate simulated data by choice
sim_choice1_rt = sim_rt(sim_choice == 1);
sim_choice2_rt = sim_rt(sim_choice == 2);

% Create a 2x1 subplot layout
subplot(2,1,1);
hold on;

% Plot density for choice == 1
[f_real1, xi_real1] = ksdensity(real_choice1_rt);
plot(xi_real1, f_real1, 'b-', 'LineWidth', 2);

[f_sim1, xi_sim1] = ksdensity(sim_choice1_rt);
plot(xi_sim1, f_sim1, 'r--', 'LineWidth', 2);

title('RT Distributions for Choice = 1');
xlabel('Response Time (RT)');
ylabel('Density');
legend('Real Data', 'Simulated Data');

% Plot density for choice == 2
subplot(2,1,2);
hold on;

[f_real2, xi_real2] = ksdensity(real_choice2_rt);
plot(xi_real2, f_real2, 'b-', 'LineWidth', 2);

[f_sim2, xi_sim2] = ksdensity(sim_choice2_rt);
plot(xi_sim2, f_sim2, 'r--', 'LineWidth', 2);

title('RT Distributions for Choice = 2');
xlabel('Response Time (RT)');
ylabel('Density');
legend('Real Data', 'Simulated Data');

%% examine 
df = table(ptp_id, real_choice, sim_choice, real_rt, sim_rt);

% Reconstruct trials:
% Find the unique participant IDs
unique_ptp_ids = unique(df.ptp_id);
% Initialize the trial column with zeros
df.trial = zeros(height(df), 1);
% Iterate through each participant and assign trial numbers
for i = 1:length(unique_ptp_ids)
    current_ptp_id = unique_ptp_ids(i);
    % Find rows for this participant
    participant_rows = df.ptp_id == current_ptp_id;
    % Assign sequential trial numbers to these rows
    df.trial(participant_rows) = (1:sum(participant_rows))';
end

disp('Mean accurate model predictions of choice:')
mean(df.real_choice==df.sim_choice)

%% Claude, please help do trial-by-trial trajectory...
% Calculate summary statistics grouped by trial
stats_table = groupsummary(df, 'trial', {'mean', 'std'},...
    {'real_rt', 'sim_rt', 'real_choice', 'sim_choice'});

figure;
subplot(2,1,1)
hold on;
% Extract the trial numbers and statistics + Plot with error bars
trials = stats_table.trial;
mean_real = stats_table.mean_real_rt;
std_real = stats_table.std_real_rt;
mean_sim = stats_table.mean_sim_rt;
std_sim = stats_table.std_sim_rt;
errorbar(trials, mean_real, std_real, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
errorbar(trials, mean_sim, std_sim, 'r-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
% Add labels and legend
xlabel('Trial');
ylabel('Reaction Time');
title('Mean Reaction Time by Trial: Real vs Simulated');
legend('Real RT', 'Simulated RT', 'Location', 'best');
% Make the plot nicer
set(gca, 'FontSize', 12);
box on;
hold off

subplot(2,1,2)
hold on
mean_real_choice = stats_table.mean_real_choice;
std_real_choice = stats_table.std_real_choice;
mean_sim_choice = stats_table.mean_sim_choice;
std_sim_choice = stats_table.std_sim_choice;
errorbar(trials, mean_real_choice, std_real_choice, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
errorbar(trials, mean_sim_choice, std_sim_choice, 'r-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
% Add labels and legend
xlabel('Trial');
ylabel('Choice (proportion)');
title('Mean Choice by Trial: Real vs Simulated');
legend('Real Choice', 'Simulated Choice', 'Location', 'best');

% Make the plot nicer
set(gca, 'FontSize', 12);
box on;
hold off

%% check correlation between simulated and real data

figure;
subplot(2,2,1)
% real_logRT = arrayfun(@(x) mean(log(x.rt)), data);
% sim_logRT = arrayfun(@(x) mean(log(x.rt)), sim_data);
mvis_scat_ref_corr(arrayfun(@(x) mean(x.rt), data), arrayfun(@(x) mean(x.rt), sim_data))
xlabel('Actual mean reaction time (s)')
ylabel('Simulated mean reaction time (s)')
subplot(2,2,2)
mvis_scat_ref_corr(arrayfun(@(x) mean(x.c), data),...
    arrayfun(@(x) mean(x.c), sim_data))
xlabel('Actual choice bias')
ylabel('Simulated choice bias')
subplot(2,2,3)
mvis_scat_ref_corr(arrayfun(@(x) calc_switch_rate(x.c), data),...
    arrayfun(@(x) calc_switch_rate(x.c), sim_data))
xlabel('Actual switch rate')
ylabel('Simulated switch rate')
subplot(2,2,4)
mvis_scat_ref_corr(arrayfun(@(x) calc_data_loseshift(x), data),...
    arrayfun(@(x) calc_data_loseshift(x), sim_data))
xlabel('Actual lose-shift')
ylabel('Simulated lose-shift')

set(gcf,'Position',[440 353 468 445])
