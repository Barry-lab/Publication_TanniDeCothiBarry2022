addpath(genpath('~/Documents/MATLAB/BVCSR/V1/_BasicFunctions'))
addpath(genpath('~/Documents/MATLAB/BVCSR/V1/_Universal_Matlab-master'))

% fixed_distance = true or false toggles the distance between place cells
% to constant (true) or to be more densely distributed at boundaries (false)
% with Tannie et al place cell axis squashing (orthogonal to boundaries)
fixed_distance = true;

dim_x = 400;
dim_y = 300;

% big environment
polys = cell(0);
polys{1} = [25 25, 25 275, 375 275, 375 25, 25 25];

env = GenerateEnv(polys, dim_x, dim_y);

if (fixed_distance == true)
    x_field = 0:10:400;
    y_field = 0:10:300;
else
    x_distance_rule = (0:(sqrt(dim_x/2)/20):sqrt(dim_x/2)).^2;
    y_distance_rule = (0:(sqrt(dim_y/2)/15):sqrt(dim_y/2)).^2;

    x_field = [x_distance_rule, dim_x-fliplr(x_distance_rule(1:end-1))];
    y_field = [y_distance_rule, dim_y-fliplr(y_distance_rule(1:end-1))];
end

cells = generateTanniPCs(env,x_field, y_field);
n_cells = length(cells);
cells = cells./sum(cells,1);
stacked_map = squeeze(sum(cells,1));

if fixed_distance == true
    example_cell_ids = [748, 668];
else
    example_cell_ids = [752, 668];
end
figure
hold on
set(gcf,'color','w')
hold on
subplot(2,1,1)
imagesc(squeeze(cells(example_cell_ids(1),:,:)))
if (fixed_distance == true)
    title('example fixed distance place cells')
else
    title('example tanni et al. place cells')
end
colormap jet
daspect([1 1 1])
axis off
subplot(2,1,2)
imagesc(squeeze(cells(example_cell_ids(2),:,:)))
colormap jet
daspect([1 1 1])
axis off

figure
hold on
title('total firing')
imagesc(stacked_map)
colormap jet
% colorbar
caxis([0,1])
daspect([1 1 1])
axis off
set(gcf,'color','w')

% calculate rate of change in pop vector
state_ids = GetStateIds(env);
n_bins = sum(env.L == 2,'all');
rate_change = nan(4,env.dim_y,env.dim_x);

for i = 1:n_bins
    [y,x] = find(state_ids == i);
    pop_vector = cells(:,y,x);

    if (~isnan(state_ids(y+1,x)))
        next_pop_vector = cells(:,y+1,x);
        rate_change(1,y,x) = norm(next_pop_vector - pop_vector);
    end
    if (~isnan(state_ids(y-1,x)))
        next_pop_vector = cells(:,y-1,x);
        rate_change(2,y,x) = norm(next_pop_vector - pop_vector);
    end
    if (~isnan(state_ids(y,x+1)))
        next_pop_vector = cells(:,y,x+1);
        rate_change(3,y,x) = norm(next_pop_vector - pop_vector);
    end
    if (~isnan(state_ids(y,x-1)))
        next_pop_vector = cells(:,y,x-1);
        rate_change(4,y,x) = norm(next_pop_vector - pop_vector);
    end
end

direction = ["north","south","east","west"];

figure
set(gcf,'color','w')
hold on
subplot(2,2,1)
imagesc(squeeze(rate_change(1,:,:)))
title('rate change north')
colormap jet
caxis([0,max(rate_change(1,:,:),[],'all')])
daspect([1 1 1])
axis off
subplot(2,2,2)
imagesc(squeeze(rate_change(2,:,:)))
title('rate change south')
colormap jet
caxis([0,max(rate_change(2,:,:),[],'all')])
daspect([1 1 1])
axis off
subplot(2,2,3)
imagesc(squeeze(rate_change(3,:,:)))
title('rate change east')
colormap jet
caxis([0,max(rate_change(3,:,:),[],'all')])
daspect([1 1 1])
axis off
subplot(2,2,4)
imagesc(squeeze(rate_change(4,:,:)))
title('rate change west')
colormap jet
caxis([0,max(rate_change(4,:,:),[],'all')])
daspect([1 1 1])
axis off


figure
imagesc(squeeze(nanmean(rate_change,1)))
title('average rate change')
colormap jet
caxis([0,max(nanmean(rate_change,1),[],'all')])
daspect([1 1 1])
axis off
set(gcf,'color','w')
