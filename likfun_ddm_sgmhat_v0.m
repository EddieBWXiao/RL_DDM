function [lik, latents] = likfun_ddm_sgmhat_v0(x,data)

% likelihood function for sigmoid-hat DDM
% modified from likfun_bandit

% strangely, likfun_binary decided to free up b0 not w
% and strangely, w cannot be recovered; so....

kappa = x(1);
a = x(2);
v0 = x(3); % drift bias
T = x(4); % ndt 

    % initialization
    lik = 0; 
    data.rt = max(eps,data.rt - T);
    
    for n = 1:data.N
        
        % data for current trial
        c = data.c(n);              % choice ("1 or not 1" coded)
        s = data.s(n);              % stimulus
        
        % drift rate
        v = v0 + kappa*s;
        
        % accumulate log-likelihod
        if c == 1
            v = -v; 
        end
        P = wfpt(data.rt(n), -v, a, 0.5);  % Wiener first passage time distribution
        
        if isnan(P) || P==0; P = realmin; end % avoid NaNs and zeros in the logarithm
        
        lik = lik + log(P);
        
        % store latent variables
        if nargout > 1
            latents.v(n,1) = v;
            latents.P(n,1) = 1/(1+exp(-a*v));
            latents.RT_mean(n,1) = (0.5*a/v)*tanh(0.5*a*v)+T;
        end
        
    end


end