%% PSO vs GA with Robust GA History (Reviewer-Fixed)
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
PSO_all = zeros(maxIter,numRuns);
GA_all  = nan(maxIter,numRuns);   % GA may stop early

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

    %% =============== GA =================
    GA_best = nan(maxIter,1);   % exact size

    numVars = 2*numSensors;
    lb = zeros(1,numVars);
    ub = ones(1,numVars);
    IntCon = 1:numSensors;

    options = optimoptions('ga', ...
        'PopulationSize',30, ...
        'MaxGenerations',maxIter, ...
        'Display','off', ...
        'OutputFcn',@(opt,state,flag) gaOutFixed(opt,state,flag));

    ga(@(z) gaFitnessFixed(z,E_i,A_i,R_i,E_max,Safety_req,R_min), ...
       numVars,[],[],[],[],lb,ub,[],IntCon,options);

    % SAFE assignment (never size-mismatched)
    GA_best = evalin('base','GA_best');
    GA_all(:,run) = GA_best(1:maxIter);
end

%% ================= PLOT (Mean ± Std) =================
PSO_mean = mean(PSO_all,2);
PSO_std  = std(PSO_all,0,2);

GA_mean  = mean(GA_all,2,'omitnan');
GA_std   = std(GA_all,0,2,'omitnan');

figure; hold on;
plot(PSO_mean,'LineWidth',2);
plot(GA_mean,'LineWidth',2);

fill([1:maxIter fliplr(1:maxIter)], ...
     [PSO_mean'+PSO_std' fliplr(PSO_mean'-PSO_std')], ...
     'b','FaceAlpha',0.2,'EdgeColor','none');

fill([1:maxIter fliplr(1:maxIter)], ...
     [GA_mean'+GA_std' fliplr(GA_mean'-GA_std')], ...
     'r','FaceAlpha',0.2,'EdgeColor','none');

xlabel('Iteration');
ylabel('Fitness');
legend('PSO','GA','Location','best');
grid on;
title('PSO vs GA (Mean ± Std over 30 runs)');

%% ================= FUNCTIONS =================
function f = fitnessFixed(x,T,E,A,R,Emax,Sreq,Rmin)
    energy = sum(x .* T .* E);
    safety = sum(x .* A .* T);

    idx = (x == 1);
    if any(idx)
        resp = min(R(idx) .* T(idx));   % WORST-CASE
    else
        resp = 0; %resp= responsiveness  
    end

    P_energy = max(0, energy - Emax);
    P_safety = max(0, Sreq - safety).^2;
    P_resp   = 10 * max(0, Rmin - resp).^2;

    f = energy + P_energy + P_safety + P_resp;
end

function f = gaFitnessFixed(z,E,A,R,Emax,Sreq,Rmin)
    n = length(E);
    x = round(z(1:n));
    T = z(n+1:end);
    f = fitnessFixed(x,T,E,A,R,Emax,Sreq,Rmin);
end

function [state,options,optchanged] = gaOutFixed(options,state,flag)
    optchanged = false;

    switch flag
        case 'init'
            GA_best = nan(options.MaxGenerations,1);
            assignin('base','GA_best',GA_best);

        case 'iter'
            GA_best = evalin('base','GA_best');
            gen = state.Generation;   % starts at 0

            if gen < options.MaxGenerations
                GA_best(gen+1) = min(state.Score);
            end

            assignin('base','GA_best',GA_best);
    end
end
