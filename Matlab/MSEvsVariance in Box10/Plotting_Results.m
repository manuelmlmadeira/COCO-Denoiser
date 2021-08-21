 clear all; close all; clc

figure
hold on
legend
% SGD
load('SGD/SGD_MSE')
poly_list_SGD = zeros(K_max,3);
Expectation_MSE_list = mean(MSE_list, 3);
for K = 1:K_max
    x = sigma_list;
    y = Expectation_MSE_list(K,:)/K;
    [p, S] = polyfix(x.', y, 1, 0, 0); 
    R2 = 1 - (S.normr/norm(y - mean(y)))^2;
    poly_list_SGD(K, :) = [p R2];
    color = [0.7 0.7 (1 - K/(K_max))];
    plot(x, y, 'DisplayName', strcat('SGD', num2str(K)), 'LineWidth', 3, 'color', color)
end

for K = 1:K_max
    p = poly_list_SGD(K,1:2);
    plot(x, polyval(p, x),'r-.', 'LineWidth', 1) 
end

lgd = legend('SGD_1','SGD_2','SGD_3','SGD_4','SGD_5','SGD_6','SGD_7','SGD_8','SGD_9','SGD_{10}','Location','northwest')
lgd.FontSize = 14;
xlabel('\sigma^2', 'FontSize', 18)
ylabel('MSE(g_k)', 'FontSize', 18)

xlim([sigma_list(1) sigma_list(end)])
ylim([0 3.5*10^3])

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];

fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

print('-bestfit', fig, 'SGD-MSEvsSigma','-dpdf')


figure
hold on
% Co-coercivity
load('Co-coercivity/CC_MSE')
poly_list_CC = zeros(K_max,3);
Expectation_MSE_list = mean(MSE_list, 3);
for K = 1:K_max
    x = sigma_list;
    y = Expectation_MSE_list(K,:)/K;
    [p, S] = polyfix(x.', y, 1, 0, 0);
    R2 = 1 - (S.normr/norm(y - mean(y)))^2;
    poly_list_CC(K, :) = [p R2];
    color = [0.7 0.7 (1 - K/(K_max))];
    plot(x, y,  'DisplayName', strcat('COCO_', num2str(K)), 'LineWidth', 3, 'color', color)  
end

for K = 1:K_max
    p = poly_list_CC(K,1:2);
    plot(x, polyval(p, x),'r-.','LineWidth', 1) 
end

lgd = legend('COCO_1','COCO_2','COCO_3','COCO_4','COCO_5','COCO_6','COCO_7','COCO_8','COCO_9','COCO_{10}','Location','northwest')
lgd.FontSize = 14;
xlabel('\sigma^2', 'FontSize', 18)
ylabel('MSE(\theta_k)', 'FontSize', 18)

xlim([sigma_list(1) sigma_list(end)])
ylim([0 3.5*10^3])

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];

fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

print('-bestfit', fig, 'COCO-MSEvsSigma','-dpdf')




% Slopes
figure
plot(1:K_max, poly_list_SGD(:,1), '-o', 'LineWidth', 1)
hold on 
plot(1:K_max, poly_list_CC(:,1), '-o', 'LineWidth', 1)
lgd = legend('SGD', 'COCO')
lgd.FontSize = 14;
xlabel('K', 'FontSize', 18)
ylabel('Slope', 'FontSize', 18)
ylim([0 4])

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];

fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

print('-bestfit', fig, 'Slopes-MSEvsSigma','-dpdf')