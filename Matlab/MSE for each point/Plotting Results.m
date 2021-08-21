%% Plot each config
clear all; close all; clc;
file_loaded = 'results2'
% results1 - box10
% results2 - box100
% results3 - box1000
load(file_loaded)

for config_index = 1:n_configurations
    figure
    hold on
    theoretical_oracle = n * sigma^2;
    plot(1:n_points_considered, theoretical_oracle * ones(1, n_points_considered),'k--', 'LineWidth', 1)
    errorbar(1:n_points_considered, MSE_singular_raw(config_index,:), STD_singular_raw(config_index,:), ':', 'LineWidth', 1)
    errorbar(1:n_points_considered, MSE_singular_denoised(config_index,:), STD_singular_denoised(config_index,:), ':', 'LineWidth', 1)
    lgd = legend('Oracle (T)', 'Oracle (E)', 'COCO', 'Location', 'southeast')
    lgd.FontSize = 14;
    xlabel('x_k', 'FontSize', 18)
    ylabel('MSE(\theta_k)', 'FontSize', 18)
    ylim([0 theoretical_oracle+50])
    xlim([0,n_points_considered+1])
    ax.XTick = 1:8;
    grid on
    
    %xlim([1 8])
    ylim([0 350])
    
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

    print('-bestfit', fig, strcat('MSE_eachpoint','_', file_loaded, '_', num2str(config_index)),'-dpdf')
    
end

%% Plot 3d of each config
h = figure
visited_points_1 = visited_points_history(:,:,3).';
x = visited_points_1(:,1);
y = visited_points_1(:,2);
z = visited_points_1(:,3);

sizes = 200*(x + 150)/200.';
colors = [[0.75 0.75 0]; [1 0 1]; [0 1 1]; [1 0 0]; [0 1 0]; [0 0 1]; [0 0 0]; [0.5 0.5 0.5]];

scatter3(x, y, z, sizes, colors, 'filled')
xlabel('X');
ylabel('Y');
zlabel('Z');
for pt_idx = 1:n_points_considered
    text(x(pt_idx), y(pt_idx), z(pt_idx)+10, strcat('x_', num2str(pt_idx)), 'FontSize', 14)
end
l = 100;
set(gca,'XLim',[-l l],'YLim',[-l l],'ZLim',[-l l])

%set(h,'Units','Inches');
%pos = get(h,'Position');
%set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
%print(h,'3D config','-dpdf','-r0')
