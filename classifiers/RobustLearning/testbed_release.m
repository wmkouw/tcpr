% This script is for illustration purpose
% Last modified: July 28, 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
clear
clc

addingPath;

nRuns = 10;
lossRobTrAd = zeros(nRuns, 1);
lossNonRobTrAd = zeros(nRuns, 1);
lossRobTeAd = zeros(nRuns, 1);
lossNonRobTeAd = zeros(nRuns, 1);
lossNonRobTe = zeros(nRuns, 1);
lossRobTe = zeros(nRuns, 1);
lossKLIEP = zeros(nRuns, 1);
lossKMM = zeros(nRuns, 1);
lossClust = zeros(nRuns, 1);
lossRuLSIF = zeros(nRuns, 1);
optSigma = zeros(nRuns, 1);
kernel = 'linear';
switch kernel
    case 'linear'
        options.kernel = @linearkernel;
        options.learner_sigma = 1;
    case 'polynomial'
        options.kernel = @polykernel;
        options.learner_sigma = 2;
    case 'gaussian'
        options.kernel = @gausskernel;
        options.learner_sigma = 0.2;
end

%% Synthetic
t = 100;
onest = ones(t, 1);
model = 'cubic';
switch model
    case 'linear'
        f = @(x)(x + 1);
    case 'cubic'
        f = @(x)(- x + x.^3 + 1);
end
beta = 0.01;
noise = 0.1;
options.useGamma = true; % swtich between robust learning and RCSA
options.beta = beta;
options.C = 1/(t*options.beta);
options.type = 'R';
for i = 1:nRuns
    rng('default');
    rng(i*10);
%     rng('shuffle','twister');
    x1 = randn(t,1)/2 + 0.5;
    y1 = f(x1) + noise*randn(t,1);
    x2 = 0.3*randn(t,1);
    y2 = f(x2) + noise*randn(t,1);
    Xtr = [x1,onest];
    ytr = y1;
    Xte = [x2,onest];
    yte = y2;
    [lossAd, loss, errAd, err, theta, sigma, weight] =...
        robust_experiment(Xtr, Xte, ytr, yte, options);
    if options.useGamma
        optSigma(i) = sigma.rob;
        lossNonRobTe(i) = loss.nonRobTe;
        lossRobTe(i) = loss.robTe;
        lossKMM(i) = loss.KMM;
        lossKLIEP(i) = loss.KLIEP;
        lossClust(i) = loss.Clust;
        lossRuLSIF(i) = loss.RuLSIF;
    else
        lossRobTrAd(i) = lossAd.robTr;
        lossNonRobTrAd(i) = lossAd.nonRobTr;
        lossRobTeAd(i) = lossAd.robTe;
        lossNonRobTeAd(i) = lossAd.nonRobTe;
        optSigma(i) = sigma.rob;
    end
end

if options.useGamma
    m = [mean(lossNonRobTe), mean(lossClust), mean(lossKMM), mean(lossKLIEP),...
        mean(lossRuLSIF), mean(lossRobTe)]';
    s = [std(lossNonRobTe), std(lossClust), std(lossKMM), std(lossKLIEP),...
        std(lossRuLSIF), std(lossRobTe)]';
    fg = figure;
    hold on
    h = bar(m,'c');
    errorbar(m,s,'k','linestyle','none');
    set(gca,'XTick', 1:6);
    XLabels = {'Unweighed','Clust','KMM','KLIEP','RuLSIF','RCSA'};
    set(gca,'XTickLabel',XLabels);
    title(['Test losses of different algorithms for ',model,' example']);
%     print(fg,'-depsc',['Figure/fig_toy_numerical_',model,'.eps']);
else
    if ttest2(lossRobTeAd, lossNonRobTeAd, 0.05)
        display('Noticeably misspecified');
    else
        display('Relatively well-specified');
    end
end

%% Auto-mpg
load('dataset/Auto-mpg.mat');
Xtr = X1;
Xte = X3; % or X2
ytr = y1;
yte = y3; % or y2
t = size(Xtr,1);
te = size(Xte,1);
Xtr = [ones(t,1),Xtr];
Xte = [ones(te,1),Xte];

rng('default');
rng(1000);
% rng('shuffle','twister');
rp1 = randperm(t);
rp2 = randperm(te);
Xtr = Xtr(rp1,:);
ytr = ytr(rp1);
Xte = Xte(rp2,:);
yte = yte(rp2);

options.type = 'R';
% use properly cross-validated parameters
switch kernel
    case 'linear'
        options.C = 100;
    case 'quadratic'
        options.C = 0.005;
        options.learner_sigma = 2;
    case 'gaussian'
        options.C = 10;
        options.learner_sigma = 10;
end
options.useGamma = false;
part = floor(t/nRuns);
for i = 1:nRuns
    display(['Running the ',num2str(i),' pack...'])
    X1 = Xtr;
    y1 = ytr;
    X1(((i-1)*part+1):(i*part),:) = [];
    y1(((i-1)*part+1):(i*part)) = [];
    [lossAd, loss, errAd, err, theta, sigma, weight] =...
        robust_experiment(X1, Xte, y1, yte, options);
    if options.useGamma
        optSigma(i) = sigma.rob;
        lossNonRobTe(i) = loss.nonRobTe;
        lossRobTe(i) = loss.robTe;
        lossKMM(i) = loss.KMM;
        lossKLIEP(i) = loss.KLIEP;
        lossClust(i) = loss.Clust;
        lossRuLSIF(i) = loss.RuLSIF;
    else
        lossRobTrAd(i) = lossAd.robTr;
        lossNonRobTrAd(i) = lossAd.nonRobTr;
        lossRobTeAd(i) = lossAd.robTe;
        lossNonRobTeAd(i) = lossAd.nonRobTe;
        optSigma(i) = sigma.rob;
    end
end
if options.useGamma
    m = [mean(lossNonRobTe), mean(lossClust), mean(lossKMM), mean(lossKLIEP),...
        mean(lossRuLSIF), mean(lossRobTe)]';
    s = [std(lossNonRobTe), std(lossClust), std(lossKMM), std(lossKLIEP),...
        std(lossRuLSIF), std(lossRobTe)]';
    fg = figure;
    hold on
    h = bar(m,'c');
    errorbar(m,s,'k','linestyle','none');
    set(gca,'XTick', 1:6);
    XLabels = {'Unweighed','Clust','KMM','KLIEP','RuLSIF','RCSA'};
    set(gca,'XTickLabel',XLabels);
    title('Test losses of different algorithms for Auto-mpg');
%     print(fg,'-depsc',['Figure/fig_toy_numerical_',model,'.eps']);
else
    if ttest2(lossRobTeAd, lossNonRobTeAd, 0.05)
        display('Noticeably misspecified');
    else
        display('Relatively well-specified');
    end
end
