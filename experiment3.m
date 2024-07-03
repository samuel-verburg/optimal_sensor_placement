% Fixed frequency, varying budget
clear
addpath('functions')

soundSpeed = 343;
f = 860; % frequency for optimization
wn = 2*pi*f/soundSpeed;

% load candiates
[rA,rB,m,mB,rx,ry] = load_candidates(0);

%% Wave model 
load('./data/uniform_sampled_sphere_200.mat')
w = V; clear V
n = size(w,1);

A = exp(1i*wn*rA*w');
B = exp(1i*wn*rB*w');

SNR = 20;
beta = 1;
alpha = n*beta / 10^(SNR/10);

%% Sensor selection
kMax = 200; % sensor budget
kVec = 5:5:kMax;

% random distribution
ran.N = 200;
for j = 1:ran.N
    iSel = randperm(m,kMax);
    ran.iSel(:,j) = iSel(:);
    ran.z(:,j) = zeros(m,1);
    ran.z(iSel,j) = 1;
end

% POD
% Generate training data
rng(21)
nTrain = 1e4;
xTrain = sqrt(alpha*2)^-1 * (randn(n,nTrain) + 1j*randn(n,nTrain));
e = sqrt(beta*2)^-1 *(randn(m,nTrain) + 1j*randn(m,nTrain));
yTrain = A*xTrain + e;

[pod.U,pod.S,pod.V] = svd(yTrain,'econ');
pod.S = diag(pod.S);
% figure
% semilogy(pod.S)
pod.nPod = 150;
pod.U = pod.U(:,1:pod.nPod);
[~,~,pod.iSel] = qr(pod.U*pod.U','vector');
pod.iSel = pod.iSel';

threshold = 0.9;

% distributions as a function of sensor budget
for  i = 1:length(kVec)
    k = kVec(i);

    % optimized distributions - convex relaxation
    % wrt to x
    [optX.iSel{i,1}, optX.zHat(:,i)] = min_Finv(A, [], k, beta, alpha, [], [], []);
    [zHat,ind] = sort(optX.zHat(:,i), 'descend');
    ind = ind( (cumsum(zHat)/sum(zHat)) < threshold );
    optX.iSel{i,1} = min_Finv_greedy(A(ind,:), [], k, beta, alpha, []);
    optX.iSel{i,1} = ind(optX.iSel{i,1});

    [optX.varX(i), optX.varBx(i)] = get_crb(optX.iSel{i,1}, A, B, beta, alpha);
    [optX.varHatX(i), optX.varHatBx(i)] = get_crb_z(optX.zHat(:,i), A, B, beta, alpha);
    
    % wrt Bx
    [optBx.iSel{i,1}, optBx.zHat(:,i)] = min_Finv(A, B, k, beta, alpha, [], [], []);
    [zHat,ind] = sort(optBx.zHat(:,i), 'descend');
    ind = ind( (cumsum(zHat)/sum(zHat)) < threshold );
    optBx.iSel{i,1} = min_Finv_greedy(A(ind,:), B, k, beta, alpha, []);
    optBx.iSel{i,1} = ind(optBx.iSel{i,1});

    [optBx.varX(i), optBx.varBx(i)] = get_crb(optBx.iSel{i,1}, A, B, beta, alpha);
    [optBx.varHatX(i), optBx.varHatBx(i)] = get_crb_z(optBx.zHat(:,i), A, B, beta, alpha);

    % pod
    [pod.varX(i), pod.varBx(i)] = get_crb(pod.iSel(1:k), A, B, beta, alpha);

    % random
    for j = 1:ran.N
        [varX(j), varBx(j)] = get_crb(ran.iSel(1:k,j), A, B, beta, alpha);
    end
    % mean and std over realizations of the random distribution
    [ran.varX(i), ran.varXStd(i)] = mean_and_std(varX);
    [ran.varBx(i), ran.varBxStd(i)] = mean_and_std(varBx);

end

%% Save and load data
% save('./data/experiment3.mat')
% load('./data/experiment3.mat')

%% Figures
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',12)

colors = [0.83 0.14 0.14
             1.00 0.54 0.00
             0.09 0.74 0.81
             0.47 0.25 0.80
             0.25 0.80 0.54];        

lw = 2;

ran.nrmseX = 100*sqrt(ran.varX*alpha/n);
optX.nrmseX = 100*sqrt(optX.varX*alpha/n);
optBx.nrmseX = 100*sqrt(optBx.varX*alpha/n);
optX.nrmseXHat = 100*sqrt(optX.varHatX*alpha/n);
optBx.nrmseXHat = 100*sqrt(optBx.varHatX*alpha/n);
pod.nrmseX = 100*sqrt(pod.varX*alpha/n);


ran.nrmseBx = 100*sqrt(ran.varBx*alpha/(n*mB));
optX.nrmseBx = 100*sqrt(optX.varBx*alpha/(n*mB));
optBx.nrmseBx = 100*sqrt(optBx.varBx*alpha/(n*mB));
optX.nrmeBxHat = 100*sqrt(optX.varHatBx*alpha/(n*mB));
optBx.nrmeBxHat = 100*sqrt(optBx.varHatBx*alpha/(n*mB));
pod.nrmseBx = 100*sqrt(pod.varBx*alpha/(n*mB));


figure('Units','normalized', 'Position',[0 0 0.2 0.7])
subplot(2,1,1)
plot(kVec, ran.nrmseX, 'Color',colors(2,:), 'Linewidth',lw)
hold on
plot(kVec, pod.nrmseX, 'Color',colors(3,:), 'Linewidth',lw)
plot(kVec, optX.nrmseX, 'Color',colors(4,:), 'Linewidth',lw)
plot(kVec, optX.nrmseXHat, 'Color',[0.8 0.8 0.8], 'Linewidth',lw)

stdPlotUp = 100*sqrt((ran.varX + ran.varXStd)*alpha/n);
stdPlotDown = 100*sqrt((ran.varX - ran.varXStd)*alpha/n);
kVec2 = [kVec, fliplr(kVec)];
inBetween = [stdPlotUp, fliplr(stdPlotDown)];
fill(kVec2, inBetween,colors(2,:), 'FaceAlpha',0.5, 'EdgeColor','none');
xticks([0:40:200])
ylim([59 100])
legend('ran','pod','$\mathrm{opt}_\mathbf{x}$','$\mathrm{tr}\mathbf{F}^{-1} (\hat{\mathbf{z}})$','Location','northeast')
xlabel('sensor budget $k$')
ylabel('nrmse($\mathbf{x}$) \%')
grid on 
title('(a)')

subplot(2,1,2)
plot(kVec, ran.nrmseBx, 'Color',colors(2,:), 'Linewidth',lw)
hold on
plot(kVec, pod.nrmseBx, 'Color',colors(3,:), 'Linewidth',lw)
plot(kVec, optBx.nrmseBx, 'Color',colors(5,:), 'Linewidth',lw)
plot(kVec, optBx.nrmeBxHat, 'Color',[0.8 0.8 0.8], 'Linewidth',lw)

stdPlotUp = 100*sqrt((ran.varBx + ran.varBxStd)*alpha/(n*mB));
stdPlotDown = 100*sqrt((ran.varBx - ran.varBxStd)*alpha/(n*mB));
kVec2 = [kVec, fliplr(kVec)];
inBetween = [stdPlotUp, fliplr(stdPlotDown)];
fill(kVec2, inBetween,colors(2,:), 'FaceAlpha',0.5, 'EdgeColor','none');
xticks([0:40:200])
legend('ran','pod','$\mathrm{opt}_\mathbf{Bx}$','$\mathrm{tr}\mathbf{BF}^{-1} (\hat{\mathbf{z}}) \mathbf{B}^\mathrm{H}$','Location','northeast')
xlabel('sensor budget $k$')
ylabel('nrmse($\mathbf{Bx}$) \%')
grid on
title('(b)')

% saveas(gcf,'./figures/exp3_fig1','epsc')