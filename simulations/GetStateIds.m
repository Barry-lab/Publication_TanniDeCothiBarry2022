function state_ids= GetStateIds(env)
%GETSTATEIDS Summary of this function goes here
%   Detailed explanation goes here
state_ids = reshape(cumsum(reshape(env.L == 2,[],1)),env.dim_y,env.dim_x);
state_ids(env.L ~= 2) = nan;
end

