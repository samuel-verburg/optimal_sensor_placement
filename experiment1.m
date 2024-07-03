% Sensor distributions
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

%% Save/load data
% save('./data/experiment1.mat')
% load('./data/experiment1.mat')

%% Figures
sSize = 10;

figure('Units','normalized', 'Position',[0 0 0.45 1])
subplot(3,2,1)
scatter(rA(:,1),rA(:,2),sSize,'k', 'MarkerEdgeAlpha',0.2)
hold on 
scatter(rA(uni.iSel,1),rA(uni.iSel,2),sSize+2,'r','filled')
title('(a) uniform')
axis equal
ylim([-0.9 0.9])
xlim([-1.3 1.3])
xlabel('(m)')
ylabel('(m)')

subplot(3,2,2)
scatter(rA(:,1),rA(:,2),sSize,'k', 'MarkerEdgeAlpha',0.2)
hold on 
scatter(rA(pod.iSel,1),rA(pod.iSel,2),sSize+2,'r','filled')
title('(b) pod')
axis equal
ylim([-0.9 0.9])
xlim([-1.3 1.3])
xlabel('(m)')
ylabel('(m)')

subplot(3,2,3)
scatter(rA(:,1),rA(:,2),sSize,'k', 'MarkerEdgeAlpha',0.2)
hold on 
scatter(rA(optX.iSel,1),rA(optX.iSel,2),sSize+2,'r','filled')
title('(c) $\mathrm{opt}_\mathbf{x}$')
axis equal
ylim([-0.9 0.9])
xlim([-1.3 1.3])
xlabel('(m)')
ylabel('(m)')

subplot(3,2,4)
scatter(rA(:,1),rA(:,2),sSize,'k', 'MarkerEdgeAlpha',0.2)
hold on 
scatter(rA(optBx.iSel,1),rA(optBx.iSel,2),sSize+2,'r','filled')
title('(d) $\mathrm{opt}_\mathbf{Bx}$')
axis equal
ylim([-0.9 0.9])
xlim([-1.3 1.3])
xlabel('(m)')
ylabel('(m)')

subplot(3,2,5)
scatter(rA(:,1),rA(:,2),sSize,'k', 'MarkerEdgeAlpha',0.2)
hold on 
scatter(rA(:,1),rA(:,2),sSize,optX.zHat, 'filled')
title('(e) $\hat{\mathbf{z}}_\mathbf{x}$')
axis equal
ylim([-0.9 0.9])
xlim([-1.3 1.3])
xlabel('(m)')
ylabel('(m)')

subplot(3,2,6)
scatter(rA(:,1),rA(:,2),sSize,'k', 'MarkerEdgeAlpha',0.2)
hold on 
scatter(rA(:,1),rA(:,2),sSize,optBx.zHat, 'filled')
title('(f) $\hat{\mathbf{z}}_\mathbf{Bx}$')
axis equal
ylim([-0.9 0.9])
xlim([-1.3 1.3])
xlabel('(m)')
ylabel('(m)')
cbar = colorbar;
cbar.Position = [0.92, 0.125, 0.0247, 0.185];
caxis([0 1])

colormap(sky)

% saveas(gcf,'./figures/exp1_fig1','epsc')
