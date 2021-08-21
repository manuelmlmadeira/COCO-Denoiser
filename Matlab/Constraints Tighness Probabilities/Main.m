clear all; clc; close all;

sigma = 10;
L_real = 1;
L_list = [0.1, 0.55, 1, 1.45, 1.9];
delta_x_list = 0:1:120;
p_inactive_list = zeros(length(L_list), length(delta_x_list));

figure
hold on
legend
for L_idx = 1:length(L_list);
    p_inactive = zeros(1, length(delta_x_list));
    L = L_list(L_idx);
    delta_L = L - L_real;
    L
    for delta_x_idx = 1: length(delta_x_list)
        delta_x = delta_x_list(delta_x_idx);
        Phi_1 = normcdf(delta_L * delta_x/(sqrt(2)*sigma));
        Phi_2 = normcdf(- L_real * delta_x/(sqrt(2)*sigma));
        p_inactive(delta_x_idx) = Phi_1 - Phi_2;
    end
    p_inactive_list(L_idx,:) = p_inactive;
    plot(delta_x_list, p_inactive, '-', 'LineWidth', 1, 'DisplayName', strcat('\Delta_L= ', num2str(delta_L)))
end
legend('Location','eastoutside')
legend show
xlabel('\Delta_x')
ylabel('p_{inactive}')
ylim([0  1])
hold off
grid on

figure
p_active_list = zeros(size(p_inactive_list));
hold on
legend
for L_idx = 1:length(L_list)
    p_active = 1-p_inactive_list(L_idx, :);
    p_active_list(L_idx,:) = p_active;
    L = L_list(L_idx);
    delta_L = L - L_real;
    plot(delta_x_list, p_active, '-', 'LineWidth', 1, 'DisplayName', strcat('\Delta_L=', num2str(delta_L)))
end
lgd = legend('Location','northeast')
lgd.FontSize = 14;
legend show
xlabel('\Delta_x', 'FontSize', 18)
ylabel('p_{active}', 'FontSize', 18)
ylim([0  1])
grid on

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

print('-bestfit', fig, 'Constraints_Tightness_Distance','-dpdf')



% Sigma Effect
figure
hold on
for L_idx = 1:2:length(L_list)
    set(gca,'ColorOrderIndex', L_idx)
    p_active = p_active_list(L_idx, :);
    L = L_list(L_idx);
    delta_L = L - L_real;
    plot(delta_x_list, p_active, '-', 'LineWidth', 0.5, 'DisplayName', strcat('\sigma=10', '; \Delta_L= ', num2str(delta_L)))
end

sigma = 20;
legend
for L_idx = 1:2:length(L_list);
    set(gca,'ColorOrderIndex', L_idx)
    p_active = zeros(1, length(delta_x_list));
    L = L_list(L_idx);
    delta_L = L - L_real;
    for delta_x_idx = 1: length(delta_x_list)
        delta_x = delta_x_list(delta_x_idx);
        Phi_1 = normcdf(delta_L * delta_x/(sqrt(2)*sigma));
        Phi_2 = normcdf(- L_real * delta_x/(sqrt(2)*sigma));
        p_active(delta_x_idx) = 1 - (Phi_1 - Phi_2);
    end
    p_active_list(L_idx,:) = p_active;
    plot(delta_x_list, p_active, '-.', 'LineWidth', 1, 'DisplayName', strcat('\sigma=20','; \Delta_L= ', num2str(delta_L)))
end

lgd = legend('Location','northeast')
lgd.FontSize = 14;
legend show
xlabel('\Delta_x', 'FontSize', 18)
ylabel('p_{active}', 'FontSize', 18)
ylim([0  1])
hold off
grid on 

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

print('-bestfit', fig, 'Constraints_Tightness_Noise','-dpdf')
