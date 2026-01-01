    %% PSO vs Heuristic Baselines (Fixed & Random)
clear; clc; close all;

%% ================= PARAMETERS =================
numSensors   = 5;
numParticles = 30;
maxIter      = 100;
numRuns      = 30;

% PSO parameters
w = 0.7; 
c1 = 1.5; 
c2 = 1.5;

% System parameters
E_i = [1.2 0.8 1.0 0.9 1.5];
A_i = [0.9 0.8 0.7 0.85 0.75];
R_i = [0.8 0.85 0.7 0.9 0.75];

E_max      = 10;
Safety_req = 3.5;
R_min      = 0.85;   % activates responsiveness

%% ================= STORAGE =================
PSO_all    = zeros(maxIter,numRuns);
Fixed_all  = zeros(maxIter,numRuns);
Random_all = zeros(maxIter,numRuns);

%% ================= MULTI-RUN LOOP =================
for run = 1:numRuns
    rng(run);

    %% =============== PSO =================
    x  = randi([0 1],numParticles,numSensors);
    T  = rand(numParticles,numSensors);
    vx = randn(numParticles,numSensors);
    vT = randn(numParticles,numSensors);

    pbest_x = x;
    pbest_T = T;
    pbest_f = inf(numParticles,1);

    fitFun = @(x,T) fitnessFixed(x,T,E_i,A_i,R_i,E_max,Safety_req,R_min);

    for i=1:numParticles
        pbest_f(i) = fitFun(x(i,:),T(i,:));
    end

    PSO_best = zeros(maxIter,1);

    for it = 1:maxIter
        for i=1:numParticles
            f = fitFun(x(i,:),T(i,:));
            if f < pbest_f(i)
                pbest_f(i) = f;
                pbest_x(i,:) = x(i,:);
                pbest_T(i,:) = T(i,:);
            end
        end

        [PSO_best(it),gidx] = min(pbest_f);
        gx = pbest_x(gidx,:);
        gT = pbest_T(gidx,:);

        r1 = rand; r2 = rand;
        vx = w*vx + c1*r1*(pbest_x-x) + c2*r2*(gx-x);
        vT = w*vT + c1*r1*(pbest_T-T) + c2*r2*(gT-T);

        x = double(x+vx > 0.5);
        T = min(max(T+vT,0),1);
    end

    PSO_all(:,run) = PSO_best;

    %% =============== Fixed-Pattern Heuristic =================
    fixed_x = repmat([1 0 1 0 1], maxIter, 1);      % Example fixed pattern
    fixed_T = repmat([0.8 0.6 0.9 0.7 0.85], maxIter,1); % Duty cycles
    Fixed_best = zeros(maxIter,1);
    for it = 1:maxIter
        Fixed_best(it) = fitFun(fixed_x(it,:), fixed_T(it,:));
    end
    Fixed_all(:,run) = Fixed_best;

    %% =============== Random-Pattern Heuristic =================
    Random_best = zeros(maxIter,1);
    for it = 1:maxIter
        rand_x = randi([0 1],1,numSensors);
        rand_T = rand(1,numSensors);
        Random_best(it) = fitFun(rand_x, rand_T);
    end
    Random_all(:,run) = Random_best;

end

%% ================= PLOT (Mean ± Std) =================
PSO_mean    = mean(PSO_all,2);     PSO_std    = std(PSO_all,0,2);
Fixed_mean  = mean(Fixed_all,2);   Fixed_std  = std(Fixed_all,0,2);
Random_mean = mean(Random_all,2);  Random_std = std(Random_all,0,2);

figure; hold on;
plot(PSO_mean,'LineWidth',2);
plot(Fixed_mean,'LineWidth',2);
plot(Random_mean,'LineWidth',2);

fill([1:maxIter fliplr(1:maxIter)], ...
     [PSO_mean'+PSO_std' fliplr(PSO_mean'-PSO_std')], ...
     'b','FaceAlpha',0.2,'EdgeColor','none');

fill([1:maxIter fliplr(1:maxIter)], ...
     [Fixed_mean'+Fixed_std' fliplr(Fixed_mean'-Fixed_std')], ...
     'g','FaceAlpha',0.2,'EdgeColor','none');

fill([1:maxIter fliplr(1:maxIter)], ...
     [Random_mean'+Random_std' fliplr(Random_mean'-Random_std')], ...
     'r','FaceAlpha',0.2,'EdgeColor','none');

xlabel('Iteration'); ylabel('Fitness');
legend('PSO','Fixed','Random','Location','best');
grid on;
title('PSO vs Fixed & Random Heuristic Baselines (Mean ± Std over 30 runs)');

%% ================= FITNESS FUNCTION =================
function f = fitnessFixed(x,T,E,A,R,Emax,Sreq,Rmin)
    energy = sum(x .* T .* E);
    safety = sum(x .* A .* T);

    idx = (x == 1);
    if any(idx)
        resp = min(R(idx) .* T(idx));   % WORST-CASE
    else
        resp = 0; 
    end

    P_energy = max(0, energy - Emax);
    P_safety = max(0, Sreq - safety).^2;
    P_resp   = 10 * max(0, Rmin - resp).^2;

    f = energy + P_energy + P_safety + P_resp;
end
