function visualize_map_pcd(P1, P2, T12,confidence, angles1, angles2)

set(gcf,'color','w');
if nargin <= 3
    confidence = 50*ones(length(P1), 1);
end

if nargin<5
    angles1 = [0, 90]; 
    angles2 = angles1;
end

g1 = P2(:, 1);
g2 = P2(:, 2);
g3 = P2(:, 3);

g1 = normalize_function(0,1,g1);
g2 = normalize_function(0,1,g2);
g3 = normalize_function(0,1,g3);

f1 = g1(T12);
f2 = g2(T12);
f3 = g3(T12);

point_size = 50; 


% subplot(1, 3, 1);
% scatter3(P1(:, 1), P1(:, 2), P1(:, 3), point_size, 'r', 'filled');
% axis equal; grid off;  title('source')%hold on; 
% view(angles1); 
%subplot(1, 2, 1); 
%imagesc(C)
subplot(1, 2, 1); 
scatter3(P2(:, 1), P2(:, 2), P2(:, 3), point_size, [g1, g2, g3], 'filled'); 
% trimesh(P2.surface.TRIV, X2, Y2, Z2, ...
%     'FaceVertexCData', [g1 g2 g3], 'FaceColor','interp', ...
%     'FaceAlpha', 1, 'EdgeColor', 'none'); 
axis equal; grid off; %hold on;
axis off; 
view(angles2); 
subplot(1, 2, 2); 
% trimesh(P1.surface.TRIV, X1, Y1, Z1, ...
%     'FaceVertexCData', [f1 f2 f3], 'FaceColor','interp', ...
%     'FaceAlpha', 1, 'EdgeColor', 'none');
scatter3(P1(:, 1), P1(:, 2), P1(:, 3),confidence, [f1, f2, f3],'filled');
%scatter3(P1(:, 1), P1(:, 2), P1(:, 3),confidence, [f1, f2, f3],'.');
axis equal; grid off;%hold on; 
axis off; 
view(angles1); 

end

function fnew = normalize_function(min_new,max_new,f)
    fnew = f - min(f);
    fnew = (max_new-min_new)*fnew/max(fnew) + min_new;
end

% function fnew = normalize_function(min_new, max_new, f)
%     fnew = f - min(f);
%     fnew = (max_new - min_new) * fnew / max(fnew) + min_new;
%     % 增加对比度
%     fnew = imadjust(fnew, [min(fnew), max(fnew)], [0, 1]);
% end