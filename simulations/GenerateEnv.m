function [ env ] = GenerateEnv( polys, dim_x, dim_y )
%GENERATEENV Summary of this function goes here
%   takes the polygons

map = zeros(dim_y+1,dim_x+1);

for i = 1:length(polys)
    map = insertShape(map,'Line',polys{i},'Color','r');
    env.polys{i} = reshape(polys{i},2,length(polys{i})/2)';
end
map = logical(map(:,:,1));
L = bwlabel(~map,4); % label regions
%map(L==2) = 1; % walls and outside = 1

dwmap = map;
dwmap = bwdist(dwmap);
dwmap(L==1) = NaN;

env.map = map; %map
env.dwmap = dwmap; %distance to wall map
env.dim_x = dim_x+1;
env.dim_y = dim_y+1;
env.L = L;

figure
imagesc(~map);
colormap gray
axis xy on
title('Image of environment','FontWeight','normal')

end

