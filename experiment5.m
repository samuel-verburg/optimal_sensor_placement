% Experiment with measured data
% Requires to download the dataset https://doi.org/10.11583/DTU.13315289.v1
% see README
clear
addpath('functions')

soundSpeed = 343;
f = 860; % frequency for optimization
wn = 2*pi*f/soundSpeed;

% load candiates
[rA,rB,m,mB,rx,ry,indIn] = load_candidates(0);
% load data
[H, r, fVec] = load_room_data();

yB = H(indIn,:);
y = H(~indIn,:);

%% Wave model 
load('./data/uniform_sampled_sphere_200.mat')
w = V; clear V
n = size(w,1);

A = exp(1i*wn*rA*w');
B = exp(1i*wn*rB*w');
C = exp(1i*wn*r*w');

%% Approx SNR
SNR = 10;
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
ran.iSel = randperm(m,k);
ran.z = zeros(m,1);
ran.z(ran.iSel,1) = 1;

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

for i = 1:length(fVec)

    disp(['frequency ' num2str(i) ' out of ' num2str(length(fVec))])

    wn = 2*pi*fVec(i)/soundSpeed;
    A = exp(1i*wn*rA*w');
    B = exp(1i*wn*rB*w');

    % uniform
    [uni.varX(i), uni.varBx(i), Finv] = get_crb(uni.iSel, A, B, beta, alpha);
    uni.x(:,i) = beta*Finv*(A(uni.iSel,:)'*y(uni.iSel,i));
    uni.Bx(:,i) = B*uni.x(:,i);
    uni.error(i) = sqrt((norm(B*uni.x(:,i) - yB(:,i)))^2 / norm(yB(:,i))^2);
    uni.simil(i) = get_similarity(B*uni.x(:,i),yB(:,i));

    % random
    [ran.varX(i), ran.varBx(i), Finv] = get_crb(ran.iSel, A, B, beta, alpha);
    ran.x(:,i) = beta*Finv*(A(ran.iSel,:)'*y(ran.iSel,i));
    ran.Bx(:,i) = B*ran.x(:,i);
    ran.error(i) = sqrt((norm(B*ran.x(:,i) - yB(:,i)))^2 / norm(yB(:,i))^2);
    ran.simil(i) = get_similarity(B*ran.x(:,i),yB(:,i));

    % optimized for x
    [optX.varX(i), optX.varBx(i), Finv] = get_crb(optX.iSel, A, B, beta, alpha);
    optX.x(:,i) = beta*Finv*(A(optX.iSel,:)'*y(optX.iSel,i));
    optX.Bx(:,i) = B*optX.x(:,i);
    optX.error(i) = sqrt((norm(B*optX.x(:,i) - yB(:,i)))^2 / norm(yB(:,i))^2);
    optX.simil(i) = get_similarity(B*optX.x(:,i),yB(:,i));

    % optimized for Bx
    [optBx.varX(i), optBx.varBx(i), Finv] = get_crb(optBx.iSel, A, B, beta, alpha);
    optBx.x(:,i) = beta*Finv*(A(optBx.iSel,:)'*y(optBx.iSel,i));
    optBx.Bx(:,i) = B*optBx.x(:,i);
    optBx.error(i) = sqrt((norm(B*optBx.x(:,i) - yB(:,i)))^2 / norm(yB(:,i))^2);
    optBx.simil(i) = get_similarity(B*optBx.x(:,i),yB(:,i));

    % pod
    [pod.varX(i), pod.varBx(i), Finv] = get_crb(pod.iSel, A, B, beta, alpha);
    pod.x(:,i) = beta*Finv*(A(pod.iSel,:)'*y(pod.iSel,i));
    pod.Bx(:,i) = B*pod.x(:,i);
    pod.error(i) = sqrt((norm(B*pod.x(:,i) - yB(:,i)))^2 / norm(yB(:,i))^2);
    pod.simil(i) = get_similarity(B*pod.x(:,i),yB(:,i));
    
end

%% Save and load data
% save('./data/experiment5.mat')
% load('./data/experiment5.mat')

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
nMov = 50;

figure('Units','normalized', 'Position',[0 0 0.2 0.7])
subplot(2,1,1)
plot(fVec, movmean(uni.error*100,nMov), 'Color',colors(1,:), 'Linewidth',lw)
hold on
plot(fVec, movmean(ran.error*100,nMov), 'Color',colors(2,:), 'Linewidth',lw)
plot(fVec, movmean(pod.error*100,nMov), 'Color',colors(3,:), 'Linewidth',lw)
plot(fVec, movmean(optX.error*100,nMov), 'Color',colors(4,:), 'Linewidth',lw)
plot(fVec, movmean(optBx.error*100,nMov), 'Color',colors(5,:), 'Linewidth',lw)
xlabel('frequency (Hz)')
ylabel('error($\mathbf{Bx}$) \%')
grid on
ylim([0 100])
xlim([50 1e3])
yticks([0:20:100])
title('(a)')
legend('uni','rand','pod','$\mathrm{opt}_\mathbf{x}$','$\mathrm{opt}_\mathbf{Bx}$','Location','northwest')

subplot(2,1,2)
plot(fVec, movmean(uni.simil*100,nMov), 'Color',colors(1,:), 'Linewidth',lw)
hold on
plot(fVec, movmean(ran.simil*100,nMov), 'Color',colors(2,:), 'Linewidth',lw)
plot(fVec, movmean(pod.simil*100,nMov), 'Color',colors(3,:), 'Linewidth',lw)
plot(fVec, movmean(optX.simil*100,nMov), 'Color',colors(4,:), 'Linewidth',lw)
plot(fVec, movmean(optBx.simil*100,nMov), 'Color',colors(5,:), 'Linewidth',lw)
xlabel('frequency (Hz)')
ylabel('similarity($\mathbf{Bx}$) \%')
grid on
ylim([0 100])
xlim([50 1e3])
yticks([0:20:100])
title('(b)')

% saveas(gcf,'./figures/exp5_fig1','epsc')

%%
iB = find(round(rB(:,1),2)==0.5 & round(rB(:,2),2)==0);
figure('Units','normalized', 'Position',[0 0 0.7 0.5])
subplot(2,1,1)
plot(fVec, 20*log10(abs(yB(iB,:))), 'k', 'Linewidth',lw+2)
hold on
plot(fVec, 20*log10(abs(optBx.Bx(iB,:))),'Color',colors(5,:), 'Linewidth',lw)
grid on 
ylim([-60 0])
xlim([50 1e3])
xlabel('frequency (Hz)')
ylabel('room response (dB)')
title('$\mathrm{(a)\ reconstruction\ at\ centre\ of\ right\ ellipse\ - opt}_\mathbf{Bx}$')
legend('reference','$\mathrm{opt}_\mathbf{Bx}$','Location','southeast')

subplot(2,1,2)
plot(fVec, 20*log10(abs(yB(iB,:))), 'k', 'Linewidth',lw+2)
hold on
plot(fVec, 20*log10(abs(pod.Bx(iB,:))),'Color',colors(3,:), 'Linewidth',lw)
grid on 
ylim([-60 0])
xlim([50 1e3])
xlabel('frequency (Hz)')
ylabel('room response (dB)')
title('(b) reconstruction at centre of right ellipse - pod')
legend('reference','pod','Location','southeast')

% saveas(gcf,'./figures/exp5_fig2','epsc')

%%
iF = find(fVec==f);
figure('Units','normalized', 'Position',[0 0 0.48 0.69])
subplot(3,2,1)
scatter(rB(:,1), rB(:,2), [], abs(yB(:,iF)), 'filled')
axis equal
caxis([min(abs(yB(:,iF))) max(abs(yB(:,iF)))])
cbar = colorbar;
cbar.Label.Interpreter = 'latex';
cbar.Label.String = '$|\mathbf{y}_\mathbf{B}|$';
title('(a) Reference sound field at 860 Hz')
xlabel('(m)')
ylabel('(m)')

subplot(3,2,3)
scatter(rB(:,1), rB(:,2), [], abs(optBx.Bx(:,iF)), 'filled')
axis equal
caxis([min(abs(yB(:,iF))) max(abs(yB(:,iF)))])
cbar = colorbar;
cbar.Label.Interpreter = 'latex';
cbar.Label.String = '$|\mathbf{B\tilde{x}}|$';
title('$\mathrm{(b)\ Reconstruction\ - opt}_\mathbf{Bx}$')
xlabel('(m)')
ylabel('(m)')

subplot(3,2,5)
scatter(rB(:,1), rB(:,2), [], abs(pod.Bx(:,iF)), 'filled')
axis equal
caxis([min(abs(yB(:,iF))) max(abs(yB(:,iF)))])
cbar = colorbar;
cbar.Label.Interpreter = 'latex';
cbar.Label.String = '$|\mathbf{B\tilde{x}}|$';
title('$\mathrm{(c)\ Reconstruction\ - pod}$')
xlabel('(m)')
ylabel('(m)')

subplot(3,2,4)
scatter(rB(:,1), rB(:,2), [], abs(yB(:,iF)-optBx.Bx(:,iF)), 'filled')
axis equal
caxis([min(abs(yB(:,iF))) max(abs(yB(:,iF)))])
cbar = colorbar;
cbar.Label.Interpreter = 'latex';
cbar.Label.String = '$|\mathbf{y}_\mathbf{B}-\mathbf{B\tilde{x}}|$';
title('$\mathrm{(d)\ Error\ - opt}_\mathbf{Bx}$')
xlabel('(m)')
ylabel('(m)')

subplot(3,2,6)
scatter(rB(:,1), rB(:,2), [], abs(yB(:,iF)-pod.Bx(:,iF)), 'filled')
axis equal
caxis([min(abs(yB(:,iF))) max(abs(yB(:,iF)))])
colormap(jet)
cbar = colorbar;
cbar.Label.Interpreter = 'latex';
cbar.Label.String = '$|\mathbf{y}_\mathbf{B}-\mathbf{B\tilde{x}}|$';
title('$\mathrm{(e)\ Error\ - pod}$')
xlabel('(m)')
ylabel('(m)')

% saveas(gcf,'./figures/exp5_fig3','epsc')