% set parameters
n_trials = 10000;
v = 1.0;
a = 1.5;
z = 0.5;
ndt = 0.2;
opts.max_t = 10;
opts.dt = 0.001;

% Generate simulated data
[rt_sim, choice_sim] = sim_wiener_diffusion([v a z ndt],...
    n_trials,...
    opts);

% Remove non-decision time for PDF calculation
rt_decision = rt_sim - ndt;

% Create time points for PDF calculation
t_range = linspace(0.01, max(rt_decision), 100);

% Calculate PDFs
pdf_upper = wfpt(t_range, -v, a, z); %???? there is a sign-flip here???
pdf_upper_sim = wfpt(sort(rt_decision(choice_sim==1)), v, a, z);

% figure;
% Upper boundary
subplot(2,1,1)
histogram(rt_decision(choice_sim==1), 40, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
plot(t_range, pdf_upper, 'b-', 'LineWidth', 2);
title('Upper Boundary Responses');
xlabel('Decision Time (s)');
ylabel('Density');
legend('Simulation', 'wfpt PDF');
subplot(2,1,2)
histogram(rt_decision(choice_sim==1), 40, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
plot(sort(rt_decision(choice_sim==1)), pdf_upper_sim, 'b-', 'LineWidth', 2);
title('Upper Boundary Responses');
xlabel('Decision Time (s)');
ylabel('Density');
legend('Simulation', 'wfpt PDF on sim');


