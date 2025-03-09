% Example usage:
% Define parameter ranges
param_ranges = struct();
param_ranges.v  = [-2, 2];    % Drift rate
param_ranges.a  = [0.8, 2];   % Threshold
param_ranges.w  = [0.3, 0.7]; % Starting point bias
param_ranges.t0 = [0.2, 0.4]; % Non-decision time

% Run 20 parameter recovery iterations with 500 trials each
results = parameter_recovery(1000, 500, param_ranges);

% BX 20250308: there is a sign error; drift rate and starting point were
% recovered to be negatives of the true value

function results = parameter_recovery(n_iterations, n_trials, param_ranges)
    % PARAMETER_RECOVERY Perform parameter recovery for the Wiener diffusion model
    %
    % USAGE: results = parameter_recovery(n_iterations, n_trials, param_ranges)
    %
    % INPUTS:
    %   n_iterations - number of parameter recovery iterations to perform
    %   n_trials     - number of trials to simulate in each iteration
    %   param_ranges - struct with fields v, a, w, t0, each containing
    %                  [min, max] for parameter generation
    %
    % OUTPUTS:
    %   results      - struct with fields for true and recovered parameters
    
    % Default parameter ranges if not provided
    if nargin < 3
        param_ranges.v  = [-3, 3];     % Drift rate
        param_ranges.a  = [0.5, 2.5];  % Threshold
        param_ranges.w  = [0.2, 0.8];  % Starting point bias
        param_ranges.t0 = [0.1, 0.5];  % Non-decision time
    end
    
    % Initialize results structure
    results = struct();
    results.true_v  = zeros(n_iterations, 1);
    results.true_a  = zeros(n_iterations, 1);
    results.true_w  = zeros(n_iterations, 1);
    results.true_t0 = zeros(n_iterations, 1);
    results.recovered_v  = zeros(n_iterations, 1);
    results.recovered_a  = zeros(n_iterations, 1);
    results.recovered_w  = zeros(n_iterations, 1);
    results.recovered_t0 = zeros(n_iterations, 1);
    results.exitflag = zeros(n_iterations, 1);
    
    % Set optimization options
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    
    % Perform parameter recovery iterations
    for i = 1:n_iterations
        fprintf('Running iteration %d of %d...\n', i, n_iterations);
        
        % Generate random true parameters within specified ranges
        true_v  = unifrnd(param_ranges.v(1), param_ranges.v(2));
        true_a  = unifrnd(param_ranges.a(1), param_ranges.a(2));
        true_w  = unifrnd(param_ranges.w(1), param_ranges.w(2));
        true_t0 = unifrnd(param_ranges.t0(1), param_ranges.t0(2));
        
        % Store true parameters
        results.true_v(i)  = true_v;
        results.true_a(i)  = true_a;
        results.true_w(i)  = true_w;
        results.true_t0(i) = true_t0;
        
        % Simulate data with true parameters
        [rt, choice] = sim_ddm(n_trials, true_v, true_a, true_w, true_t0);
        
        % Initial parameter guess (random within range)
        x0 = [unifrnd(param_ranges.v(1), param_ranges.v(2)), ...
              unifrnd(param_ranges.a(1), param_ranges.a(2)), ...
              unifrnd(param_ranges.w(1), param_ranges.w(2)), ...
              unifrnd(param_ranges.t0(1), param_ranges.t0(2))];
        
        % Parameter bounds for optimization
        lb = [param_ranges.v(1), param_ranges.a(1), param_ranges.w(1), param_ranges.t0(1)];
        ub = [param_ranges.v(2), param_ranges.a(2), param_ranges.w(2), param_ranges.t0(2)];
        
        % Define objective function (negative log-likelihood)
        obj_fun = @(params) -diffusion_loglike(params, rt, choice);
        
        % Run optimization
        [x_recovered, ~, exitflag] = fmincon(obj_fun, x0, [], [], [], [], lb, ub, [], options);
        
        % Store recovered parameters
        results.recovered_v(i)  = x_recovered(1);
        results.recovered_a(i)  = x_recovered(2);
        results.recovered_w(i)  = x_recovered(3);
        results.recovered_t0(i) = x_recovered(4);
        results.exitflag(i) = exitflag;
    end
    
    % Calculate recovery metrics
    results.v_error  = results.recovered_v - results.true_v;
    results.a_error  = results.recovered_a - results.true_a;
    results.w_error  = results.recovered_w - results.true_w;
    results.t0_error = results.recovered_t0 - results.true_t0;
    
    results.v_correlation  = corr(results.true_v, results.recovered_v);
    results.a_correlation  = corr(results.true_a, results.recovered_a);
    results.w_correlation  = corr(results.true_w, results.recovered_w);
    results.t0_correlation = corr(results.true_t0, results.recovered_t0);
    
    % Plot results
    plot_recovery_results(results);
end

function loglike = diffusion_loglike(params, rt, choice)
    % Calculate log-likelihood of data given diffusion model parameters
    %
    % INPUTS:
    %   params - [v, a, w, t0] parameters
    %   rt     - vector of reaction times
    %   choice - vector of choices (1 = upper boundary, 0 = lower boundary)
    %
    % OUTPUTS:
    %   loglike - log-likelihood value
    
    % Extract parameters
    v  = params(1);
    a  = params(2);
    w  = params(3);
    t0 = params(4);
    
    % Initialize log-likelihood
    loglike = 0;
    
    % Minimum allowed RT density for numerical stability
    min_density = 1e-10;
    
    % Process upper boundary responses (choice == 1)
    upper_idx = (choice == 1);
    if any(upper_idx)
        upper_rt = rt(upper_idx) - t0;  % Remove non-decision time
        
        % Skip or adjust any RTs that are too small after t0 subtraction
        valid_idx = upper_rt > 0;
        if any(valid_idx)
            % Calculate density for each valid RT
            density = wfpt(upper_rt(valid_idx), v, a, w);
            
            % Clip density values to avoid log(0)
            density = max(density, min_density);
            
            % Add to log-likelihood
            loglike = loglike + sum(log(density));
        end
    end
    
    % Process lower boundary responses (choice == 0)
    lower_idx = (choice == 0);
    if any(lower_idx)
        lower_rt = rt(lower_idx) - t0;  % Remove non-decision time
        
        % Skip or adjust any RTs that are too small after t0 subtraction
        valid_idx = lower_rt > 0;
        if any(valid_idx)
            % For lower boundary, we use negative drift rate and 1-w
            density = wfpt(lower_rt(valid_idx), -v, a, 1-w);
            
            % Clip density values to avoid log(0)
            density = max(density, min_density);
            
            % Add to log-likelihood
            loglike = loglike + sum(log(density));
        end
    end
end

function plot_recovery_results(results)
    % Plot parameter recovery results
    
    % Create figure
    figure('Position', [100, 100, 1000, 800]);
    
    % Parameter names and titles
    param_names = {'v', 'a', 'w', 't0'};
    param_titles = {'Drift Rate (v)', 'Threshold (a)', 'Starting Point (w)', 'Non-decision Time (t0)'};
    
    % Loop over parameters
    for i = 1:4
        % Get true and recovered values for current parameter
        true_vals = results.(['true_' param_names{i}]);
        recovered_vals = results.(['recovered_' param_names{i}]);
        correlation = results.([param_names{i} '_correlation']);
        
        % Create scatter plot
        subplot(2, 2, i);
        scatter(true_vals, recovered_vals, 50, 'filled', 'MarkerFaceAlpha', 0.7);
        hold on;
        
        % Add identity line
        min_val = min([true_vals; recovered_vals]);
        max_val = max([true_vals; recovered_vals]);
        range = max_val - min_val;
        plot_min = min_val - range * 0.1;
        plot_max = max_val + range * 0.1;
        plot([plot_min, plot_max], [plot_min, plot_max], 'k--');
        
        % Add regression line
        b = polyfit(true_vals, recovered_vals, 1);
        plot([plot_min, plot_max], polyval(b, [plot_min, plot_max]), 'r-');
        
        % Set axis limits
        xlim([plot_min, plot_max]);
        ylim([plot_min, plot_max]);
        
        % Add labels and title
        xlabel('True Value');
        ylabel('Recovered Value');
        title(sprintf('%s (r = %.3f)', param_titles{i}, correlation));
        
        % Add grid
        grid on;
    end
    
    % Add overall title
    sgtitle('Parameter Recovery Results');
end

