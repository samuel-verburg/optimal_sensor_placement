% Fixed budget, varying frequency 
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
% uniform distribution
delta = 4;
rxDeci = rx(1:delta:end,1:delta:end);
ryDeci = ry(1:delta:end,1:delta:end);
rUni = [rxDeci(:), ryDeci(:), zeros(size(rxDeci(:)))];
[~,uni.iSel] = intersect(rA,rUni,'rows','legacy');
uni.z = zeros(m,1); 
uni.z(uni.iSel) = 1;
clear rUni rxDeci ryDeci delta 

k = length(uni.iSel); % sensor budget

% random distribution
ran.N = 200;
for j = 1:ran.N
    iSel = randperm(m,k);
    ran.iSel(:,j) = iSel(:);
    ran.z(:,j) = zeros(m,1);
    ran.z(iSel,j) = 1;
end

% optimized distributions
% wrt to x
threshold = 0.9;
[optX.iSel, optX.zHat] = min_Finv(A, [], k, beta, alpha, [], [], []);
[zHat,ind] = sort(optX.zHat, 'descend');
ind = ind( (cumsum(zHat)/sum(zHat)) < threshold );
optX.iSel = min_Finv_greedy(A(ind,:), [], k, beta, alpha, []);
optX.iSel = ind(optX.iSel);

% wrt Bx
[optBx.iSel, optBx.zHat] = min_Finv(A, B, k, beta, alpha, [], [], []);
[zHat,ind] = sort(optBx.zHat, 'descend');
ind = ind( (cumsum(zHat)/sum(zHat)) < threshold );
optBx.iSel = min_Finv_greedy(A(ind,:), B, k, beta, alpha, []);
optBx.iSel = ind(optBx.iSel);
clear zHat ind

% POD
% Generate training data
rng(21)
nTrain = 1e4;
xTrain = sqrt(alpha*2)^-1 * (randn(n,nTrain) + 1j*randn(n,nTrain));
e = sqrt(beta*2)^-1 *(randn(m,nTrain) + 1j*randn(m,nTrain));
yTrain = A*xTrain + e;

[pod.U,pod.S,pod.V] = svd(yTrain,'econ');
pod.S = diag(pod.S);
figure
semilogy(pod.S)
pod.U = pod.U(:,1:150);
[~,~,pod.iSel] = qr(pod.U*pod.U','vector');
pod.iSel = pod.iSel(1:k).';

%% compute error as a function of frequency
fMax = 1200;
fVec = 20:10:fMax;

for i = 1:length(fVec)

    disp(['frequency ' num2str(i) ' out of ' num2str(length(fVec))])

    wn = 2*pi*fVec(i)/soundSpeed;
    A = exp(1i*wn*rA*w');
    B = exp(1i*wn*rB*w');

    % uniform
    [uni.varX(i), uni.varBx(i)] = get_crb(uni.iSel, A, B, beta, alpha);

    % random
    for j = 1:ran.N
        [varX(j), varBx(j)] = get_crb(ran.iSel(:,j), A, B, beta, alpha);
    end
    % mean and std over realizations of the random distribution
    [ran.varX(i), ran.varXStd(i)] = mean_and_std(varX);
    [ran.varBx(i), ran.varBxStd(i)] = mean_and_std(varBx);

    % optimized for x
    [optX.varX(i), optX.varBx(i)] = get_crb(optX.iSel, A, B, beta, alpha);

    % optimized for Bx
    [optBx.varX(i), optBx.varBx(i)] = get_crb(optBx.iSel, A, B, beta, alpha);

    % pod
    [pod.varX(i), pod.varBx(i)] = get_crb(pod.iSel, A, B, beta, alpha);
end

%% Save/load data
% save('./data/experiment2.mat')
% load('./data/experiment2.mat')

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

uni.nrmseX = 100*sqrt(uni.varX*alpha/n);
ran.nrmseX = 100*sqrt(ran.varX*alpha/n);
optX.nrmseX = 100*sqrt(optX.varX*alpha/n);
optBx.nrmseX = 100*sqrt(optBx.varX*alpha/n);
pod.nrmseX = 100*sqrt(pod.varX*alpha/n);

uni.nrmseBx = 100*sqrt(uni.varBx*alpha/(n*mB));
ran.nrmseBx = 100*sqrt(ran.varBx*alpha/(n*mB));
optX.nrmseBx = 100*sqrt(optX.varBx*alpha/(n*mB));
optBx.nrmseBx = 100*sqrt(optBx.varBx*alpha/(n*mB));
pod.nrmseBx = 100*sqrt(pod.varBx*alpha/(n*mB));

figure('Units','normalized', 'Position',[0 0 0.2 0.7])
subplot(2,1,1)
plot(fVec, uni.nrmseX, 'Color',colors(1,:), 'Linewidth',lw)
hold on
plot(fVec, ran.nrmseX, 'Color',colors(2,:), 'Linewidth',lw)
plot(fVec, pod.nrmseX, 'Color',colors(3,:), 'Linewidth',lw)
plot(fVec, optX.nrmseX, 'Color',colors(4,:), 'Linewidth',lw)
plot(fVec, optBx.nrmseX, 'Color',colors(5,:), 'Linewidth',lw)

stdPlotUp = 100*sqrt((ran.varX + ran.varXStd)*alpha/n);
stdPlotDown = 100*sqrt((ran.varX - ran.varXStd)*alpha/n);
kVec2 = [fVec, fliplr(fVec)];
inBetween = [stdPlotUp, fliplr(stdPlotDown)];
fill(kVec2, inBetween,colors(2,:), 'FaceAlpha',0.5, 'EdgeColor','none');

title('(a)')
legend('uni','ran','pod','$\mathrm{opt}_\mathbf{x}$','$\mathrm{opt}_\mathbf{Bx}$','Location','northeast')
xlabel('frequency (Hz)')
ylabel('nrmse($\mathbf{x}$) \%')
grid on 
xticks([0:200:1200])

subplot(2,1,2)
plot(fVec, uni.nrmseBx, 'Color',colors(1,:), 'Linewidth',lw)
hold on 
plot(fVec, ran.nrmseBx, 'Color',colors(2,:), 'Linewidth',lw)
plot(fVec, pod.nrmseBx, 'Color',colors(3,:), 'Linewidth',lw)
plot(fVec, optX.nrmseBx, 'Color',colors(4,:), 'Linewidth',lw)
plot(fVec, optBx.nrmseBx, 'Color',colors(5,:), 'Linewidth',lw)

stdPlotUp = 100*sqrt((ran.varBx + ran.varBxStd)*alpha/(n*mB));
stdPlotDown = 100*sqrt((ran.varBx - ran.varBxStd)*alpha/(n*mB));
fVec2 = [fVec, fliplr(fVec)];
inBetween = [stdPlotUp, fliplr(stdPlotDown)];
fill(fVec2, inBetween,colors(2,:), 'FaceAlpha',0.5, 'EdgeColor','none');

title('(b)')
xlabel('frequency (Hz)')
ylabel('nrmse($\mathbf{Bx}$) \%')
grid on
xticks([0:200:1200])

% saveas(gcf,'./figures/exp2_fig1','epsc')