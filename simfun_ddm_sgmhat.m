function data = simfun_ddm_sgmhat(x,task,opts)
    
    % Bowen Xiao 20250309
    % forward model for likfun_ddm_sgmhat

    % unpack params
    kappa = x(1);           % drift rate differential action value weight
    a = x(2);           % decision threshold
    z = x(3); % likfun_bandit assumes no bias
    T = x(4);           % non-decision time
       
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
    
    for n = 1:data.N
        
        % determine trial-wise drift rate
        s = data.s(n);              % state
        v = kappa * s; 
        
        % response model; second arg = n_trials for sim, here is just 1
        [data.rt(n), choice] = sim_wiener_diffusion([-v, a, z, T], 1, opts);
        
        % record the choice
        data.c(n) = 2 - choice; %CAVEAT: hard-coded 1 & 0 to 1 & 2 switch
        
    end
end