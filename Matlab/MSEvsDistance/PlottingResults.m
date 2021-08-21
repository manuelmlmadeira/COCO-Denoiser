%% Plotting Results
clear all; close all; clc;
load('results2')

h = figure
subplot(2,1,1)
hold on
for L_index = 1:n_L
    delta_L = L_list(L_index) - L_real;
    plot(distance_list, p_active(:, L_index), 'LineWidth', 1, 'DisplayName', strcat('\Delta_L=', num2str(delta_L)))
end

lgd = legend('\Delta_L=-0.9', '\Delta_L=-0.45', '\Delta_L=0', '\Delta_L=0.45', '\Delta_L=0.9','Location','eastoutside')
lgd.FontSize = 14;
xlabel('\Delta_x', 'FontSize', 18)
ylabel('p_{active}', 'FontSize', 18)
grid on



subplot(2,1,2)
hold on
%legend
for L_index = 1:n_L
    delta_L = L_list(L_index) - L_real;
    plot(distance_list, MSE_total_denoised(:, L_index), 'LineWidth', 1, 'DisplayName', strcat('\Delta_L=', num2str(delta_L)))
end
plot(distance_list, n_points_considered * Sigma * ones(1,length(distance_list)), 'k--', 'LineWidth', 1, 'DisplayName', 'Oracle') %, 'LineWidth', 1, '--

lgd = legend('\Delta_L=-0.9', '\Delta_L=-0.45', '\Delta_L=0', '\Delta_L=0.45', '\Delta_L=0.9', 'Oracle','Location','eastoutside')
lgd.FontSize = 14;
xlabel('\Delta_x', 'FontSize', 18)
ylabel('MSE(\theta)', 'FontSize', 18)
ylim([0 250])
grid on

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'MSEvsDistance','-dpdf','-r0')





% BIAS Figure
h = figure
legend
for L_index = 1:n_L
    %subplot(ceil(n_L/2), 2, L_index)
    hold on
    for point_index = 1:n_points_considered
        delta_L = L_list(L_index) - L_real;
        plot(distance_list, Expected_theta(L_index,:, point_index), 'LineWidth', 1, 'DisplayName', strcat('x_', num2str(point_index), ' (\Delta_L=', num2str(delta_L), ')'))
        %plot(distance_list, Expected_raw(L_index,:, point_index), 'LineWidth', 1, 'DisplayName', strcat('x_', num2str(point_index), 'Oracle'))
    end 
    legend('Location','eastoutside')
    legend show
    xlabel('\Delta_x')
    ylabel('|| Bias(\theta) ||')
    xlim([0 120])
    ylim([0 5])
    %title(strcat('\Delta L=', num2str(delta_L)))
end
grid on

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'Bias','-dpdf','-r0')
