%% simulate
% set parameters
n_trials = 1000;
v = 1.0;
a = 1.5;
z = 0.4;
ndt = 0.2;
opts.max_t = 10;
opts.dt = 0.001;

%% Generate simulated data
rng(5)
[rt_sim, choice_sim] = sim_wiener_diffusion([v a z ndt],...
    n_trials,...
    opts);

%% SOME EXTRA PROCESSING
% Remove non-decision time for PDF calculation
rt_decision = rt_sim - ndt;
% Create time points for PDF calculation
t_range = linspace(0.01, max(rt_decision), 100);

%% Calculate PDFs
pdf_upper = wfpt(t_range, -v, a, 1-z); %???? there is a sign-flip here???
pdf_lower = wfpt(t_range, v, a, z);

%% visualise
rt_bidir = rt_decision;
rt_bidir(choice_sim==0) = -rt_bidir(choice_sim==0);

% Find the maximum absolute value to create symmetric bins
max_abs_rt = max(abs(rt_bidir));
bin_edges = linspace(-max_abs_rt, max_abs_rt, 41); % 41 edges gives 40 bins

figure;
histogram(rt_bidir, bin_edges, 'Normalization', 'pdf', 'FaceColor', 'k', 'FaceAlpha', 0.3);
hold on;
plot(t_range, pdf_upper, 'r-', 'LineWidth', 2);
plot(-t_range, pdf_lower, 'r-', 'LineWidth', 2);
hold off
legend('Simulated data','WFPT PDF')
xlabel('Decision time (s)');
ylabel('Density');

set(gcf,'Position', [440 543 378 255])

%% ------------------------------------
pdf_lower_copy = pdf_lower;
pdf_upper_copy = pdf_upper;
%% demonstrating the symmetry?
% rewrite the code above and then flip the signs
%% Generate simulated data
rng(5)
[rt_sim, choice_sim] = sim_wiener_diffusion([-v a 1-z ndt],...
    n_trials,...
    opts);

%% SOME EXTRA PROCESSING
% Remove non-decision time for PDF calculation
rt_decision = rt_sim - ndt;
% Create time points for PDF calculation
t_range = linspace(0.01, max(rt_decision), 100);

%% Calculate PDFs
pdf_lower = wfpt(t_range, -v, a, 1-z); 
pdf_upper = wfpt(t_range, v, a, z);

%% visualise
rt_bidir = rt_decision;
rt_bidir(choice_sim==0) = -rt_bidir(choice_sim==0);

% Find the maximum absolute value to create symmetric bins
max_abs_rt = max(abs(rt_bidir));
bin_edges = linspace(-max_abs_rt, max_abs_rt, 41); % 41 edges gives 40 bins

figure;
histogram(rt_bidir, bin_edges, 'Normalization', 'pdf', 'FaceColor', 'k', 'FaceAlpha', 0.3);
hold on;
plot(t_range, pdf_upper, 'r-', 'LineWidth', 2);
plot(-t_range, pdf_lower, 'r-', 'LineWidth', 2);
hold off
legend('Simulated data','WFPT PDF')
xlabel('Decision time (s)');
ylabel('Density');

set(gcf,'Position', [440 543 378 255])
%% check
mean(abs(pdf_upper_copy - pdf_lower))
mean(abs(pdf_lower_copy - pdf_upper))

figure;
plot(pdf_upper_copy,pdf_lower,'o')
hold on
plot(pdf_lower_copy,pdf_upper,'o')
refline([1 0])
hold off
