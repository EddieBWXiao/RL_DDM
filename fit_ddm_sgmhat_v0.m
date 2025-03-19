function results = fit_ddm_sgmhat_v0(data)
    
    % Fit sigmoid-hat DDM model to data.
    %
    % USAGE: results = fit_ddm_sgmhat(data)
    %
    % INPUTS:
    %   data - [S x 1] data structure, where S is the number of subjects; see likfun_ddm_sgmhat for more details
    %
    % OUTPUTS:
    %   results - see mfit_optimize for more details
    %
    % Bowen Xiao, Mar 2025, modifying from fit_bandit by Sam Gershman
    
    % create parameter structure
    
    % drift rate scaling parameter
    param(1).name = 'kappa';
    param(1).logpdf = @(x) 0;  % uniform prior
    param(1).lb = 0; % lower bound
    param(1).ub = 20;   % upper bound
    
    % decision threshold
    param(2).name = 'a';
    param(2).logpdf = @(x) 0;
    param(2).lb = 1e-3;
    param(2).ub = 20;
    
    % drift bias
    param(3).name = 'v0';
    param(3).logpdf = @(x) 0;
    param(3).lb = -20;
    param(3).ub = 20;
    
    % non-decision time
    param(4).name = 'T';
    param(4).logpdf = @(x) 0;
    param(4).lb = 0;
    param(4).ub = 1;
    
    % fit model
    f = @(x,data) likfun_ddm_sgmhat_v0(x,data);    % log-likelihood function
    results = mfit_optimize(f,param,data);