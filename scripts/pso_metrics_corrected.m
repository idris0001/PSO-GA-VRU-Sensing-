%% PSO for Energy-Efficient VRU Sensing (Corrected Responsiveness)
clear; clc; close all;

%% PSO Parameters
numParticles = 30;   % Number of particles
numSensors = 5;      % Number of sensors
maxIter = 100;       % Maximum iterations
w = 0.7;             % Inertia weight
c1 = 1.5;            % Cognitive coefficient
c2 = 1.5;            % Social coefficient

%% System Parameters
E_i = [1.2, 0.8, 1.0, 0.9, 1.5];      % Energy consumption rates
A_i = [0.9, 0.8, 0.7, 0.85, 0.75];    % Accuracy contributions
R_i = [0.8, 0.85, 0.7, 0.9, 0.75];    % Responsiveness
E_max = 10;                             % Max energy budget
Safety_req = 3.5;                       % Minimum safety requirement
R_min = 0.8;                            % Minimum responsiveness

%% Initialize Particles
x = randi([0, 1], numParticles, numSensors);  % Sensor activation (0 or 1)
T = rand(numParticles, numSensors);           % Duty cycles (0 to 1)
v_x = randn(numParticles, numSensors);        % Velocities for x
v_T = randn(numParticles, numSensors);        % Velocities for T

p_best_x = x;
p_best_T = T;
p_best_scores = inf(numParticles, 1);

%% Define fitness function as a nested function
fitnessFunction = @(x, T) ...
    energySafetyResponsivenessFitness(x, T, E_i, A_i, R_i, E_max, Safety_req, R_min);

for i = 1:numParticles
    p_best_scores(i) = fitnessFunction(x(i,:), T(i,:));
end

[~, g_best_idx] = min(p_best_scores);
g_best_x = p_best_x(g_best_idx, :);
g_best_T = p_best_T(g_best_idx, :);

%% Arrays to store metrics
energyHistory = zeros(maxIter,1);
safetyHistory = zeros(maxIter,1);
responsivenessHistory = zeros(maxIter,1);
penaltyEnergyHistory = zeros(maxIter,1);
penaltySafetyHistory = zeros(maxIter,1);
penaltyResponsivenessHistory = zeros(maxIter,1);
fitnessHistory = zeros(maxIter,1);

%% PSO Main Loop
for iter = 1:maxIter
    for i = 1:numParticles
        % Evaluate fitness
        fitness = fitnessFunction(x(i,:), T(i,:));

        % Calculate Energy, Safety, Responsiveness
        energy = sum(x(i,:) .* T(i,:) .* E_i);
        safety = sum(x(i,:) .* A_i .* T(i,:));

        % Corrected Responsiveness
        activeIdx = (x(i,:) == 1);
        if any(activeIdx)
            responsiveness = mean(R_i(activeIdx) .* T(i, activeIdx));
        else
            responsiveness = 0;
        end

        % Penalty Terms
        penaltyEnergy = max(0, energy - E_max);
        penaltySafety = max(0, Safety_req - safety);
        penaltyResponsiveness = max(0, R_min - responsiveness);

        % Store metrics
        energyHistory(iter) = energy;
        safetyHistory(iter) = safety;
        responsivenessHistory(iter) = responsiveness;
        penaltyEnergyHistory(iter) = penaltyEnergy;
        penaltySafetyHistory(iter) = penaltySafety;
        penaltyResponsivenessHistory(iter) = penaltyResponsiveness;

        % Update personal best
        if fitness < p_best_scores(i)
            p_best_x(i,:) = x(i,:);
            p_best_T(i,:) = T(i,:);
            p_best_scores(i) = fitness;
        end
    end

    % Update global best
    [~, g_best_idx] = min(p_best_scores);
    g_best_x = p_best_x(g_best_idx,:);
    g_best_T = p_best_T(g_best_idx,:);

    % Update velocities and positions
    r1 = rand(); r2 = rand();
    v_x = w*v_x + c1*r1*(p_best_x - x) + c2*r2*(repmat(g_best_x,numParticles,1)-x);
    v_T = w*v_T + c1*r1*(p_best_T - T) + c2*r2*(repmat(g_best_T,numParticles,1)-T);

    x = double(x + v_x > 0.5);        % Binary positions
    T = max(0, min(1, T + v_T));     % Clamp T to [0,1]

    % Record global best fitness
    fitnessHistory(iter) = fitnessFunction(g_best_x, g_best_T);
end

%% Plot Results
figure; plot(1:maxIter, energyHistory,'LineWidth',2);
xlabel('Iteration'); ylabel('Energy Consumption'); title('Energy Consumption Over Iterations'); grid on;

figure; plot(1:maxIter, safetyHistory,'LineWidth',2);
xlabel('Iteration'); ylabel('Safety Performance'); title('Safety Performance Over Iterations'); grid on;

figure; plot(1:maxIter, responsivenessHistory,'LineWidth',2);
xlabel('Iteration'); ylabel('Responsiveness'); title('Responsiveness Over Iterations'); grid on;

figure;
subplot(3,1,1); plot(1:maxIter, penaltyEnergyHistory,'LineWidth',2);
xlabel('Iteration'); ylabel('Energy Penalty'); title('Energy Penalty'); grid on;
subplot(3,1,2); plot(1:maxIter, penaltySafetyHistory,'LineWidth',2);
xlabel('Iteration'); ylabel('Safety Penalty'); title('Safety Penalty'); grid on;
subplot(3,1,3); plot(1:maxIter, penaltyResponsivenessHistory,'LineWidth',2);
xlabel('Iteration'); ylabel('Responsiveness Penalty'); title('Responsiveness Penalty'); grid on;

figure; plot(1:maxIter, fitnessHistory,'LineWidth',2);
xlabel('Iteration'); ylabel('Fitness'); title('Fitness Over Iterations'); grid on;

%% Nested fitness function (corrected)
function fitness = energySafetyResponsivenessFitness(x, T, E_i, A_i, R_i, E_max, Safety_req, R_min)
    energy = sum(x .* T .* E_i);
    safety = sum(x .* A_i .* T);
    activeIdx = (x == 1);
    if any(activeIdx)
        responsiveness = mean(R_i(activeIdx) .* T(activeIdx));
    else
        responsiveness = 0;
    end
    penaltyEnergy = max(0, energy - E_max);
    penaltySafety = max(0, Safety_req - safety);
    penaltyResponsiveness = max(0, R_min - responsiveness);
    fitness = energy + penaltyEnergy + penaltySafety + penaltyResponsiveness;
end
