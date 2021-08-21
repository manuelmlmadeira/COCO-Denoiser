% Plotting
clear all; close all; clc;

% SGD
load('SGD/SGD')
SGD_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_distances(:, rep) = vecnorm_local(SGD_visited(:,:,rep) - x_star, 1);
end
SGD_distances_mean = mean(SGD_distances, 2);
SGD_distances_SEM = std(SGD_distances, 0, 2)/sqrt(n_reps);

load('SGD_COCO2/COCO2')
SGD_COCO2_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_COCO2_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
SGD_COCO2_distances_mean = mean(SGD_COCO2_distances, 2);
SGD_COCO2_distances_SEM = std(SGD_COCO2_distances, 0, 2)/sqrt(n_reps);

load('SGD_COCO4/COCO4')
SGD_COCO4_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_COCO4_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
SGD_COCO4_distances_mean = mean(SGD_COCO4_distances, 2);
SGD_COCO4_distances_SEM = std(SGD_COCO4_distances, 0, 2)/sqrt(n_reps);

load('SGD_COCO8/COCO8')
SGD_COCO8_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_COCO8_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
SGD_COCO8_distances_mean = mean(SGD_COCO8_distances, 2);
SGD_COCO8_distances_SEM = std(SGD_COCO8_distances, 0, 2)/sqrt(n_reps);

load('SGD_COCO16/COCO16')
SGD_COCO16_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_COCO16_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
SGD_COCO16_distances_mean = mean(SGD_COCO16_distances, 2);
SGD_COCO16_distances_SEM = std(SGD_COCO16_distances, 0, 2)/sqrt(n_reps);

load('SGD_COCO/COCO')
SGD_COCO_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_COCO_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
SGD_COCO_distances_mean = mean(SGD_COCO_distances, 2);
SGD_COCO_distances_SEM = std(SGD_COCO_distances, 0, 2)/sqrt(n_reps);

% Adam

load('Adam/Adam')
Adam_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    Adam_distances(:, rep) = vecnorm_local(Adam_visited(:,:,rep) - x_star, 1);
end
Adam_distances_mean = mean(Adam_distances, 2);
Adam_distances_SEM = std(Adam_distances, 0, 2)/sqrt(n_reps);

load('Adam_COCO2/COCO2')
Adam_COCO2_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    Adam_COCO2_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
Adam_COCO2_distances_mean = mean(Adam_COCO2_distances, 2);
Adam_COCO2_distances_SEM = std(Adam_COCO2_distances, 0, 2)/sqrt(n_reps);

load('Adam_COCO4/COCO4')
Adam_COCO4_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    Adam_COCO4_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
Adam_COCO4_distances_mean = mean(Adam_COCO4_distances, 2);
Adam_COCO4_distances_SEM = std(Adam_COCO4_distances, 0, 2)/sqrt(n_reps);

load('Adam_COCO8/COCO8')
Adam_COCO8_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    Adam_COCO8_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
Adam_COCO8_distances_mean = mean(Adam_COCO8_distances, 2);
Adam_COCO8_distances_SEM = std(Adam_COCO8_distances, 0, 2)/sqrt(n_reps);

load('Adam_COCO16/COCO16')
Adam_COCO16_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    Adam_COCO16_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
Adam_COCO16_distances_mean = mean(Adam_COCO16_distances, 2);
Adam_COCO16_distances_SEM = std(Adam_COCO16_distances, 0, 2)/sqrt(n_reps);

load('Adam_COCO/COCO')
Adam_COCO_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    Adam_COCO_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
Adam_COCO_distances_mean = mean(Adam_COCO_distances, 2);
Adam_COCO_distances_SEM = std(Adam_COCO_distances, 0, 2)/sqrt(n_reps);


figure
iter_vec = 1:nstep_max+1
ax = axes();
set(ax, 'YScale', 'log');
hold on
errorbar(iter_vec, SGD_distances_mean, SGD_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, SGD_COCO2_distances_mean, SGD_COCO2_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, SGD_COCO4_distances_mean, SGD_COCO4_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, SGD_COCO8_distances_mean, SGD_COCO8_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, SGD_COCO16_distances_mean, SGD_COCO16_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, SGD_COCO_distances_mean, SGD_COCO_distances_SEM, 'LineWidth', 1.5)

lgd = legend('SGD', 'SGD+COCO_2', 'SGD+COCO_4', 'SGD+COCO_8', 'SGD+COCO_{16}', 'SGD+COCO', 'Location', 'northeast')
lgd.FontSize = 18;
xlabel('Oracle Consultations', 'FontSize', 18)
ylabel('E[ ||x_i - x^*||_2 ]', 'FontSize', 18)
axis tight
ylim([10 350])
hold off

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

print('-bestfit', fig,'SGD','-dpdf')


figure
ax = axes();
set(ax, 'YScale', 'log');
hold on
errorbar(iter_vec, Adam_distances_mean, Adam_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, Adam_COCO2_distances_mean, Adam_COCO2_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, Adam_COCO4_distances_mean, Adam_COCO4_distances_SEM,'LineWidth', 1.5)
errorbar(iter_vec, Adam_COCO8_distances_mean, Adam_COCO8_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, Adam_COCO16_distances_mean, Adam_COCO16_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, Adam_COCO_distances_mean, Adam_COCO_distances_SEM, 'LineWidth', 1.5)

lgd = legend('Adam', 'Adam+COCO_2', 'Adam+COCO_4', 'Adam+COCO_8', 'Adam+COCO_{16}', 'Adam+COCO', 'Location', 'northeast')
lgd.FontSize = 18;
xlabel('Oracle Consultations', 'FontSize', 18)
ylabel('E[ ||x_i - x^*||_2 ]', 'FontSize', 18)
axis tight
ylim([10 350])
hold off

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

print('-bestfit', fig,'Adam','-dpdf')

%% Plotting Tuned Step Sizes
clear all; close all; clc;


% SGD
load('SGD/SGD')
SGD_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_distances(:, rep) = vecnorm_local(SGD_visited(:,:,rep) - x_star, 1);
end
SGD_distances_mean = mean(SGD_distances, 2);
SGD_distances_SEM = std(SGD_distances, 0, 2)/sqrt(n_reps);

% COCOs
load('SGD_COCO2/COCO2')
SGD_COCO2_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_COCO2_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
SGD_COCO2_distances_mean = mean(SGD_COCO2_distances, 2);
SGD_COCO2_distances_SEM = std(SGD_COCO2_distances, 0, 2)/sqrt(n_reps);

load('SGD_COCO4/COCO4')
SGD_COCO4_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_COCO4_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
SGD_COCO4_distances_mean = mean(SGD_COCO4_distances, 2);
SGD_COCO4_distances_SEM = std(SGD_COCO4_distances, 0, 2)/sqrt(n_reps);

load('SGD_COCO8/COCO8')
SGD_COCO8_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_COCO8_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
SGD_COCO8_distances_mean = mean(SGD_COCO8_distances, 2);
SGD_COCO8_distances_SEM = std(SGD_COCO8_distances, 0, 2)/sqrt(n_reps);



% Step-size to COCO2
load('step_size_to_COCO2/SGD')
ss_COCO2_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    ss_COCO2_distances(:, rep) = vecnorm_local(SGD_visited(:,:,rep) - x_star, 1);
end
ss_COCO2_distances_mean = mean(ss_COCO2_distances, 2);
ss_COCO2_distances_SEM = std(ss_COCO2_distances, 0, 2)/sqrt(n_reps);

% Step-size to COCO4
load('step_size_to_COCO4/SGD')
ss_COCO4_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    ss_COCO4_distances(:, rep) = vecnorm_local(SGD_visited(:,:,rep) - x_star, 1);
end
ss_COCO4_distances_mean = mean(ss_COCO4_distances, 2);
ss_COCO4_distances_SEM = std(ss_COCO4_distances, 0, 2)/sqrt(n_reps);

% Step-size to COCO8
load('step_size_to_COCO8/SGD')
ss_COCO8_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    ss_COCO8_distances(:, rep) = vecnorm_local(SGD_visited(:,:,rep) - x_star, 1);
end
ss_COCO8_distances_mean = mean(ss_COCO8_distances, 2);
ss_COCO8_distances_SEM = std(ss_COCO8_distances, 0, 2)/sqrt(n_reps);




%Plotting
figure
iter_vec = 1:nstep_max+1;
ax = axes();
set(ax, 'YScale', 'log');
hold on
errorbar(iter_vec, SGD_distances_mean, SGD_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, SGD_COCO2_distances_mean, SGD_COCO2_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, SGD_COCO4_distances_mean, SGD_COCO4_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, SGD_COCO8_distances_mean, SGD_COCO8_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, ss_COCO2_distances_mean, ss_COCO2_distances_SEM,'LineWidth', 1.5)
errorbar(iter_vec, ss_COCO4_distances_mean, ss_COCO4_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, ss_COCO8_distances_mean, ss_COCO8_distances_SEM, 'LineWidth', 1.5)


lgd = legend('SGD', 'SGD+COCO_2', 'SGD+COCO_4','SGD+COCO_8','SGD+SS(COCO_2)', 'SGD+SS(COCO_4)', 'SGD+SS(COCO_8)','SGD+COCO_16', 'SGD+SS_{tuned to COCO_16}', 'Location', 'northeast')
lgd.FontSize = 18;
xlabel('Iterations', 'FontSize', 18)
ylabel('||x_i - x^*||_2', 'FontSize', 18)
axis tight
ylim([10 350])


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

print('-bestfit', fig, 'COCOvstunedSStovariance','-dpdf')

%% Polyak Ruppert Averaging and Decreasing Step Size
clear all; close all; clc;

% SGD
load('SGD/SGD')
SGD_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_distances(:, rep) = vecnorm_local(SGD_visited(:,:,rep) - x_star, 1);
end
SGD_distances_mean = mean(SGD_distances, 2);
SGD_distances_SEM = std(SGD_distances, 0, 2)/sqrt(n_reps);

% PR averaging
load('PRaveraging/PRaveraging')
PR_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    PR_distances(:, rep) = vecnorm_local(PRaveraging_visited(:,:,rep) - x_star, 1);
end
PR_distances_mean = mean(PR_distances, 2);
PR_distances_SEM = std(PR_distances, 0, 2)/sqrt(n_reps);


% Decreasing step size
load('Decreasing_SS/Decreasing_SS')
Decreasing_SS_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    Decreasing_SS_distances(:, rep) = vecnorm_local(decreasing_ss_visited(:,:,rep) - x_star, 1);
end
Decreasing_SS_mean = mean(Decreasing_SS_distances, 2);
Decreasing_SS_SEM = std(Decreasing_SS_distances, 0, 2)/sqrt(n_reps);


% COCO
load('SGD_COCO/COCO')
SGD_COCO_distances = zeros(nstep_max+1, n_reps);
for rep = 1:n_reps
    SGD_COCO_distances(:, rep) = vecnorm_local(COCO_visited(:,:,rep) - x_star, 1);
end
SGD_COCO_distances_mean = mean(SGD_COCO_distances, 2);
SGD_COCO_distances_SEM = std(SGD_COCO_distances, 0, 2)/sqrt(n_reps);

% Plotting
figure
iter_vec = 1:nstep_max +1;
ax = axes();
set(ax, 'YScale', 'log');
hold on
errorbar(iter_vec, SGD_distances_mean, SGD_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, SGD_COCO_distances_mean, SGD_COCO_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, PR_distances_mean, PR_distances_SEM, 'LineWidth', 1.5)
errorbar(iter_vec, Decreasing_SS_mean, Decreasing_SS_SEM, 'LineWidth', 1.5)

lgd = legend('SGD', 'SGD+COCO', 'SGD+PRaveraging', 'SGD+DecreasingSS', 'Location', 'northeast')
lgd.FontSize = 18;
xlabel('Oracle Consultations', 'FontSize', 18)
ylabel('E[ ||x_i - x^*||_2 ]', 'FontSize', 18)
axis tight
ylim([10 350])
hold off

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

print('-bestfit', fig, 'COCOvsOtherVR','-dpdf')
