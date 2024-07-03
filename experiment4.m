% Sparsity 
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

%% compute error as a function of sparsity
sMax = n;
sVec = 5:5:sMax;
nMC = 10;%1e3;

for i = 1:nMC
    iPerm = randperm(n,n);
    iSparse(:,i) = iPerm;
end

for i = 1:length(sVec)

    disp(['sparsity ' num2str(i) ' out of ' num2str(length(sVec))])
    s = sVec(i);
    alpha = s*beta / 10^(SNR/10);
    
    for j = 1:nMC
        % uniform
        [uni.varX(j,i), uni.varBx(j,i)] = get_crb(uni.iSel, A(:,iSparse(1:s,j)), B(:,iSparse(1:s,j)), beta, alpha);
                
        % random
        for l = 1:ran.N
            [varX(l), varBx(l)] = get_crb(ran.iSel(:,l), A(:,iSparse(1:s,j)), B(:,iSparse(1:s,j)), beta, alpha);
        end
        % mean and std over realizations of the random distribution
        [ran.varX(j,i), ran.varXStd(j,i)] = mean_and_std(varX);
        [ran.varBx(j,i), ran.varBxStd(j,i)] = mean_and_std(varBx);
        
        % optimized for x
        [optX.varX(j,i), optX.varBx(j,i)] = get_crb(optX.iSel, A(:,iSparse(1:s,j)), B(:,iSparse(1:s,j)), beta, alpha);
        
        % optimized for Bx
        [optBx.varX(j,i), optBx.varBx(j,i)] = get_crb(optBx.iSel, A(:,iSparse(1:s,j)), B(:,iSparse(1:s,j)), beta, alpha);

        % pod
        [pod.varX(j,i), pod.varBx(j,i)] = get_crb(pod.iSel, A(:,iSparse(1:s,j)), B(:,iSparse(1:s,j)), beta, alpha);
    end
end

uni.varX = mean(uni.varX);
ran.varX = mean(ran.varX);
optX.varX = mean(optX.varX);
optBx.varX = mean(optBx.varX);
pod.varX = mean(pod.varX);

uni.varBx = mean(uni.varBx);
ran.varBx = mean(ran.varBx);
optX.varBx = mean(optX.varBx);
optBx.varBx = mean(optBx.varBx);
pod.varBx = mean(pod.varBx);

ran.varXStd = mean(ran.varXStd);
ran.varBxStd = mean(ran.varBxStd);

%% Save/load data
% save('./data/experiment4.mat')
% load('./data/experiment4.mat')

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
plot(sVec/n, uni.nrmseX, 'Color',colors(1,:), 'Linewidth',lw)
hold on
plot(sVec/n, ran.nrmseX, 'Color',colors(2,:), 'Linewidth',lw)
plot(sVec/n, pod.nrmseX, 'Color',colors(3,:), 'Linewidth',lw)
plot(sVec/n, optX.nrmseX, 'Color',colors(4,:), 'Linewidth',lw)
plot(sVec/n, optBx.nrmseX, 'Color',colors(5,:), 'Linewidth',lw)

stdPlotUp = 100*sqrt((ran.varX + ran.varXStd)*alpha/(n));
stdPlotDown = 100*sqrt((ran.varX - ran.varXStd)*alpha/(n));
sVec2 = [sVec, fliplr(sVec)];
inBetween = [stdPlotUp, fliplr(stdPlotDown)];
fill(sVec2/n, inBetween,colors(2,:), 'FaceAlpha',0.5, 'EdgeColor','none');

title('(a)')
legend('uni','rand','pod','$\mathrm{opt}_\mathbf{x}$','$\mathrm{opt}_\mathbf{Bx}$','Location','southeast')
xlabel('\# non-zeros / \# parameters')
ylabel('nrmse($\mathbf{x}$) \%')
grid on 
xticks([0:0.2:1])
yticks([0:20:100])

subplot(2,1,2)
plot(sVec/n, uni.nrmseBx, 'Color',colors(1,:), 'Linewidth',lw)
hold on 
plot(sVec/n, ran.nrmseBx, 'Color',colors(2,:), 'Linewidth',lw)
plot(sVec/n, pod.nrmseBx, 'Color',colors(3,:), 'Linewidth',lw)
plot(sVec/n, optX.nrmseBx, 'Color',colors(4,:), 'Linewidth',lw)
plot(sVec/n, optBx.nrmseBx, 'Color',colors(5,:), 'Linewidth',lw)

stdPlotUp = 100*sqrt((ran.varBx + ran.varBxStd)*alpha/(n*mB));
stdPlotDown = 100*sqrt((ran.varBx - ran.varBxStd)*alpha/(n*mB));
sVec2 = [sVec, fliplr(sVec)];
inBetween = [stdPlotUp, fliplr(stdPlotDown)];
fill(sVec2/n, inBetween,colors(2,:), 'FaceAlpha',0.5, 'EdgeColor','none');

title('(b)')
xlabel('\# non-zeros / \# parameters')
ylabel('nrmse($\mathbf{Bx}$) \%')
grid on
xticks([0:0.2:1])

% saveas(gcf,'./figures/exp4_fig1','epsc')
