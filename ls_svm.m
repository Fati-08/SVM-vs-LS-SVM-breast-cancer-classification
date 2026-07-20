%% =========================================================
% LS-SVM on Breast Cancer Wisconsin
% - train / validation / test split
% - mini grid search su gamma e sigma
% - training finale e valutazione
%% =========================================================

clear; clc; close all;

%% 1) Load data
data = readtable('data.csv');

% X = 30 feature numeriche
X = table2array(data(:, 3:32));

% y = etichette binarie: M -> +1, B -> -1
% Se nel tuo file la colonna si chiama "diagnosis", usa data.diagnosis
% Se MATLAB la importa come Var2, usa data.Var2
y = double(strcmp(data.Var2, 'M')) * 2 - 1;

%% 2) Standardize features
% Porto ogni feature ad avere media 0 e deviazione standard 1
X = (X - mean(X)) ./ std(X);

%% 3) Primo split: training+validation / test
% Il test set resta separato fino alla fine
rng(42);
n = size(X,1);
idx = randperm(n);

n_trainval = round(0.8 * n);   % 80%
X_trainval = X(idx(1:n_trainval), :);
y_trainval = y(idx(1:n_trainval));

Xte = X(idx(n_trainval+1:end), :);   % 20% test
yte = y(idx(n_trainval+1:end));

%% 4) Secondo split: training / validation
% Dal blocco training+validation ricaviamo un validation set interno
n_tv = size(X_trainval,1);
idx_tv = randperm(n_tv);

n_train = round(0.8 * n_tv);   % 80% di 80% = 64% del totale
Xtr = X_trainval(idx_tv(1:n_train), :);
ytr = y_trainval(idx_tv(1:n_train));

Xval = X_trainval(idx_tv(n_train+1:end), :);   % 16% del totale
yval = y_trainval(idx_tv(n_train+1:end));

%% 5) Mini grid search su gamma e sigma
% Prove 
gamma_list = [1 10 50 100];
sigma_list = [0.5 1 2 5];

best_val_acc = 0;
best_gamma = gamma_list(1);
best_sigma = sigma_list(1);

fprintf('--- GRID SEARCH SU VALIDATION SET ---\n'); %[output:05b81e80]

for ig = 1:length(gamma_list) %[output:group:321a6eca]
    for is = 1:length(sigma_list)

        gamma = gamma_list(ig);
        sigma = sigma_list(is);

        % Train sul training set
        [b_tmp, alpha_tmp] = train_lssvm(Xtr, ytr, gamma, sigma);

        % Predict sul validation set
        y_val_pred = predict_lssvm(Xtr, ytr, Xval, b_tmp, alpha_tmp, sigma);

        % Accuracy sul validation set
        val_acc = mean(y_val_pred == yval) * 100;

        fprintf('gamma = %-5g   sigma = %-4g   validation accuracy = %.2f%%\n', ... %[output:3d2a7e9c]
                gamma, sigma, val_acc); %[output:3d2a7e9c]

        % Salvo la miglior coppia
        if val_acc > best_val_acc
            best_val_acc = val_acc;
            best_gamma = gamma;
            best_sigma = sigma;
        end
    end
end %[output:group:321a6eca]

fprintf('\nBest gamma = %g\n', best_gamma); %[output:2679261a]
fprintf('Best sigma = %g\n', best_sigma); %[output:5dbcfa11]
fprintf('Best validation accuracy = %.2f%%\n', best_val_acc); %[output:296527cf]

%% 6) Training finale sul blocco training+validation
% Ora riuso tutti i dati non-test per addestrare il modello finale
[b, alpha] = train_lssvm(X_trainval, y_trainval, best_gamma, best_sigma);

%% 7) Predict sul test set
y_pred = predict_lssvm(X_trainval, y_trainval, Xte, b, alpha, best_sigma);

%% 8) Valutazione finale
acc = mean(y_pred == yte) * 100;

TP = sum(y_pred == 1  & yte == 1);
TN = sum(y_pred == -1 & yte == -1);
FP = sum(y_pred == 1  & yte == -1);
FN = sum(y_pred == -1 & yte == 1);

precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1 = 2 * precision * recall / (precision + recall);
specificity = TN / (TN + FP);

fprintf('\n--- RISULTATI FINALI SUL TEST SET ---\n'); %[output:5e1fb6b2]
fprintf('Accuracy    : %.2f%%\n', acc); %[output:8ef9b172]
fprintf('Precision   : %.4f\n', precision); %[output:3b379ce3]
fprintf('Recall      : %.4f\n', recall); %[output:2e170c30]
fprintf('Specificity : %.4f\n', specificity); %[output:1e7f8916]
fprintf('F1-score    : %.4f\n', f1); %[output:1848cfb6]

fprintf('\nConfusion matrix:\n'); %[output:41dd69f3]
fprintf('         Pred +1   Pred -1\n'); %[output:51e36d6c]
fprintf('Reale +1   %3d       %3d\n', TP, FN); %[output:122400ff]
fprintf('Reale -1   %3d       %3d\n', FP, TN); %[output:3966246b]

%% =========================================================
% FUNCTIONS
%% =========================================================

function [b, alpha] = train_lssvm(Xtr, ytr, gamma, sigma)

    n = size(Xtr,1);

    % Matrice kernel RBF
    K = kernel_matrix(Xtr, Xtr, sigma);

    % Omega = y_i * y_j * K(x_i,x_j)
    Omega = (ytr * ytr') .* K;

    % Sistema lineare LS-SVM
    A = [0, ytr';
         ytr, Omega + (1/gamma)*eye(n)];

    rhs = [0; ones(n,1)];

    % Soluzione del sistema
    sol = A \ rhs;

    b = sol(1);
    alpha = sol(2:end);
end

function y_pred = predict_lssvm(Xtr, ytr, Xte, b, alpha, sigma)

    ntest = size(Xte,1);
    y_pred = zeros(ntest,1);

    for k = 1:ntest

        % Kernel tra tutti i punti di training e il punto di test k
        kvec = kernel_matrix(Xtr, Xte(k,:), sigma);

        % Score della LS-SVM
        score = alpha' * (ytr .* kvec) + b;

        % Classe predetta
        y_pred(k) = sign(score);

        % Se score = 0, assegno +1 per evitare ambiguità numeriche
        if y_pred(k) == 0
            y_pred(k) = 1;
        end
    end
end

function K = kernel_matrix(X1, X2, sigma)

    n1 = size(X1,1);
    n2 = size(X2,1);
    K = zeros(n1, n2);

    for i = 1:n1
        for j = 1:n2
            d = norm(X1(i,:) - X2(j,:))^2;
            K(i,j) = exp(-d / (2*sigma^2));
        end
    end
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":5.7}
%---
%[output:05b81e80]
%   data: {"dataType":"text","outputData":{"text":"--- GRID SEARCH SU VALIDATION SET ---\n","truncated":false}}
%---
%[output:3d2a7e9c]
%   data: {"dataType":"text","outputData":{"text":"gamma = 1       sigma = 0.5    validation accuracy = 59.34%\ngamma = 1       sigma = 1      validation accuracy = 67.03%\ngamma = 1       sigma = 2      validation accuracy = 97.80%\ngamma = 1       sigma = 5      validation accuracy = 97.80%\ngamma = 10      sigma = 0.5    validation accuracy = 59.34%\ngamma = 10      sigma = 1      validation accuracy = 80.22%\ngamma = 10      sigma = 2      validation accuracy = 97.80%\ngamma = 10      sigma = 5      validation accuracy = 97.80%\ngamma = 50      sigma = 0.5    validation accuracy = 59.34%\ngamma = 50      sigma = 1      validation accuracy = 81.32%\ngamma = 50      sigma = 2      validation accuracy = 96.70%\ngamma = 50      sigma = 5      validation accuracy = 96.70%\ngamma = 100     sigma = 0.5    validation accuracy = 59.34%\ngamma = 100     sigma = 1      validation accuracy = 81.32%\ngamma = 100     sigma = 2      validation accuracy = 96.70%\ngamma = 100     sigma = 5      validation accuracy = 95.60%\n","truncated":false}}
%---
%[output:2679261a]
%   data: {"dataType":"text","outputData":{"text":"\nBest gamma = 1\n","truncated":false}}
%---
%[output:5dbcfa11]
%   data: {"dataType":"text","outputData":{"text":"Best sigma = 2\n","truncated":false}}
%---
%[output:296527cf]
%   data: {"dataType":"text","outputData":{"text":"Best validation accuracy = 97.80%\n","truncated":false}}
%---
%[output:5e1fb6b2]
%   data: {"dataType":"text","outputData":{"text":"\n--- RISULTATI FINALI SUL TEST SET ---\n","truncated":false}}
%---
%[output:8ef9b172]
%   data: {"dataType":"text","outputData":{"text":"Accuracy    : 95.61%\n","truncated":false}}
%---
%[output:3b379ce3]
%   data: {"dataType":"text","outputData":{"text":"Precision   : 0.9024\n","truncated":false}}
%---
%[output:2e170c30]
%   data: {"dataType":"text","outputData":{"text":"Recall      : 0.9737\n","truncated":false}}
%---
%[output:1e7f8916]
%   data: {"dataType":"text","outputData":{"text":"Specificity : 0.9474\n","truncated":false}}
%---
%[output:1848cfb6]
%   data: {"dataType":"text","outputData":{"text":"F1-score    : 0.9367\n","truncated":false}}
%---
%[output:41dd69f3]
%   data: {"dataType":"text","outputData":{"text":"\nConfusion matrix:\n","truncated":false}}
%---
%[output:51e36d6c]
%   data: {"dataType":"text","outputData":{"text":"         Pred +1   Pred -1\n","truncated":false}}
%---
%[output:122400ff]
%   data: {"dataType":"text","outputData":{"text":"Reale +1    37         1\n","truncated":false}}
%---
%[output:3966246b]
%   data: {"dataType":"text","outputData":{"text":"Reale -1     4        72\n","truncated":false}}
%---
