function [lik, latents] = likfun_ddm_sgmhat(x,data)

% likelihood function for sigmoid-hat DDM
% modified from likfun_bandit
% strangely, likfun_binary decided to free up b0?? Need to try later

kappa = x(1);
a = x(2);
w = x(3); %basically is z
T = x(4); %ndt 

    % initialization
    lik = 0; 
    data.rt = max(eps,data.rt - T);
    
    for n = 1:data.N
        
        % data for current trial
        c = data.c(n);              % choice ("1 or not 1" coded)
        s = data.s(n);              % stimulus
        
        % drift rate
        v = kappa*s;
        
        % get log-likelihod based on choice and rt
        starting_point = a*w;
        P_vec = wfpt_prep(a, v, starting_point, data.rt(n));  % Wiener first passage time distribution
        if c == 1
            P = P_vec(1); 
        else
            P = P_vec(2);
        end
        
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