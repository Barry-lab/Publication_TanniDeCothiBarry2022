function cells = generateSanderPCs(env,x_field,y_field)
%GENERATESANDERPCS Summary of this function goes here

cells = zeros(length(x_field)*length(y_field),env.dim_y,env.dim_x);

bin_id = find(env.L == 2);
x_sig_rule = movmean(diff(x_field(1:22)),3);
y_sig_rule = movmean(diff(y_field(1:17)),3);

x_sigs = 2*[x_sig_rule, fliplr(x_sig_rule(1:end-1))];
y_sigs = 2*[y_sig_rule, fliplr(y_sig_rule(1:end-1))];
n=0;
for  i= 1:length(x_field)
    mean_x = x_field(i);
    for j = 1:length(y_field)
        n = n+1;
        mean_y = y_field(j);

        sig_x = x_sigs(i); sig_y = y_sigs(j);

        % generate rate map
        place_map = zeros(size(env.L));
        for k = 1:length(bin_id)
            [y,x] = ind2sub(size(place_map),bin_id(k));
%             if (sqrt( (x-mean_x)^2 + (y-mean_y)^2 ) < 40)
                place_map(y,x) = firingRate(x, y, mean_x, mean_y, sig_x, sig_y);
%             end
        end
%         place_map = place_map/nanmax(place_map(:));
        cells(n,:,:) = place_map;
    end
end
display(n)

end

function fr = firingRate(x,y,mean_x,mean_y,sig_x,sig_y)
fr = exp(-(x-mean_x)^2 / (2*sig_x^2)) * exp(-(y-mean_y)^2 / (2*sig_y^2)) / (2*pi*sig_x*sig_y);
end