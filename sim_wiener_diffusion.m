function [rt, choice] = sim_wiener_diffusion(x, n_trials, opts)
    % 
    % Simulate data from a Wiener diffusion model
    % Not full DDM since no RT variability, only four params
    %
    % Bowen Xiao 20250308, assisted by Claude
    % 
    % INPUTS:
        % x - parameters
            % x(1) - v: drift rate
            % x(2) - a: boundary separation
            % x(3) - z: starting point bias as proportion of 'a'; 
                % CHECK: if this is also the case for the package and for HSSM?
            % x(4) - ndt: non-decision time
        % n_trials: number of trials to simulate; in lieu of the usual "task"
        % opts - optional arguments, as a struct; fields can include:
            % dt: time step for simulation (default: 0.001 seconds)
            % max_t: maximum allowed decision time (default: 10 seconds)
    %
    % OUTPUTS:
        % rt: vector of reaction times in seconds
        % choice: vector of choices (1 = upper boundary, 0 = lower boundary)
        % NOTE for future change --> the output can include NaN if max_t exceeded on that trial
    
    %% input wrangling
    % unpack parameters
    v = x(1);
    a = x(2);
    z = x(3);
    ndt = x(4);
        
    % unpack settings, with defaults
    if ~exist('opts', 'var')
        opts = struct();
    end
    if ~isfield(opts, 'dt')
        dt = 0.001;
    else
        dt = opts.dt;
    end
    if ~isfield(opts, 'max_t')
        max_t = 10;
    else
        max_t = opts.max_t;
    end
    
    %% model
    % preallocate output
    rt = nan(n_trials, 1);
    choice = nan(n_trials, 1);
    
    % convert z to absolute
    z_absolute = z * a;
    
    % loop through trials
    for i = 1:n_trials
        
        % start at initial
        y = z_absolute;
        
        % time counter (add non-decision time later)
        drift_time = 0;
        
        % run until a boundary is reached or max time exceeded
        while (y > 0 && y < a && drift_time < max_t)
            % accumulated evidence; with noise ~N(0, 1) scaled with size of timestep
            y = y + v*dt + sqrt(dt)*randn();
            
            % keep track of the time
            drift_time = drift_time + dt;
        end
        
        % record reaction time
        rt(i) = drift_time + ndt;  % add non-decision time
        
        % generate choice by looking at which boundary was hit
        if y >= a
            choice(i) = 1;  % hit upper boundary
        elseif y <= 0
            choice(i) = 0;  % hit lower boundary
        else
            % Claude suggestion: if max time was reached, randomly assign a boundary?
            choice(i) = round(rand());
            rt(i) = max_t + ndt;
            %rt(i) = NaN; % without setting else, the choice & rt should stay as NaN!!!
        end
    end
end