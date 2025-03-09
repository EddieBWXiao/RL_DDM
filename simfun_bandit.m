function data = simfun_bandit(x,task,opts)
    
    % Bowen Xiao 20250309
    % forward model for Sam Gershman's likfun_bandit
    % TO-DO: figure out the v coding

    % unpack params
    b = x(1);           % drift rate differential action value weight
    lr = x(2);          % learning rate
    a = x(3);           % decision threshold
    T = x(4);           % non-decision time
    z = 0.5; % likfun_bandit assumes no bias
    
    % unpack settings, with defaults
    if ~exist('opts', 'var')
        opts = struct();
        opts.dt = 0.001;
        opts.max_t = 10;
    end
    
    % inherit from task
    data.C = task.C;
    data.N = task.N;
    data.s = task.s;
    % preallocate
    data.c = nan(data.N, 1);
    data.r = nan(data.N, 1);
    data.rt = nan(data.N, 1);
    
    % further preallocate
    C = data.C;
    S = length(unique(data.s)); % number of states
    Q = zeros(S, C);    % initial state-action values (Q for one trial, not as matrix to store)
    
    for n = 1:data.N
        
        % determine trial-wise drift rate
        s = data.s(n);              % state
        v = b*(Q(s,2)-Q(s,1)); %BIG BIG THINGY: here v is 2 minus 1... ??????
            % could this be the source of the v sign flip????
        
        % response model; second arg = n_trials for sim, here is just 1
        [data.rt(n), choice] = sim_wiener_diffusion([-v, a, z, T], 1, opts);
        
        % generate reward from task (currently cannot handle NaN, 1 & 0 coded)
        data.c(n) = 2 - choice; %CAVEAT: hard-coded 1 & 0 to 1 & 2 switch
        data.r(n) = task.r(n, data.c(n));
        
        % learning model:
        c = data.c(n); % choice;
        r = data.r(n); % reward
        % update values
        Q(s,c) = Q(s,c) + lr*(r - Q(s,c));
    end
end