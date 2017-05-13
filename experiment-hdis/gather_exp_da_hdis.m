% Script to gather results from domain adaptation experiments on heart disease

close all;
clearvars;

sav = false;

% Load data
dataname = 'hdis_imp0';
load(dataname)

% Experimental parameters
clfs = {'tca', 'kmm', 'rcsa', 'rba', 'tcp-ls', 'tcp-lda', 'tcp-qda'};
cc = 1:12;
nR = 1;
no = '1';
prep = 'max';

% Number of classifiers and combinations
nCl = length(clfs);
nCc = length(cc);

% Preallocate
a_r = NaN(nCl,nCc,nR);
a_e = NaN(nCl,nCc,nR);
a_a = NaN(nCl,nCc,nR);
for c = 1:length(clfs)
    for n = 1:nCc
        
        clear R e AUC
        switch clfs{c}
            case {'kmm'}
                load(['results/results_iwc_kmm_lsq_' dataname '_prep' prep '_cc' num2str(cc(n)) '_nR' num2str(nR) '_' no '.mat']);
            case {'sls','tls'}
                load(['results/results_tcp-ls_' dataname '_prep' prep '_cc' num2str(cc(n)) '_nR' num2str(nR) '_' no '.mat']);
            case {'slda','tlda'}
                load(['results/results_tcp-lda_' dataname '_prep' prep '_cc' num2str(cc(n)) '_nR' num2str(nR) '_' no '.mat']);
            case {'sqda','tqda'}
                load(['results/results_tcp-qda_' dataname '_prep' prep '_cc' num2str(cc(n)) '_nR' num2str(nR) '_' no '.mat']);
            otherwise
                load(['results/results_' clfs{c} '_' dataname '_prep' prep '_cc' num2str(cc(n)) '_nR' num2str(nR) '_' no '.mat']);
        end
        
        for r = 1:nR
            
            switch clfs{c}
                case {'sls','tcp-ls','tls','slda','tcp-lda','tlda','sqda', 'tcp-qda', 'tqda'}
                    % Average loglikelihood
                    a_r(c,n,r) = R{r};
                    
                    % Error rate
                    a_e(c,n,r) = e{r};
                    
                    % AUC
                    a_a(c,n,r) = AUC{r};
                    
                otherwise
                    % Average loglikelihood
                    a_r(c,n,r) = R(r);
                    
                    % Error rate
                    a_e(c,n,r) = e(r);
                    
                    % AUC
                    a_a(c,n,r) = AUC(r);
            end
        end
    end
end

% Compute standard error of the mean
if nR>1
    s_r = std(a_r, [], 3, 'omitnan');
    s_e = std(a_e, [], 3, 'omitnan');
    s_a = std(a_a, [], 3, 'omitnan');
else
    s_r = zeros(nCl,nCc);
    s_e = zeros(nCl,nCc);
    s_a = zeros(nCl,nCc);
end

% Average over repeats
m_r = mean(a_r, 3, 'omitnan');
m_e = mean(a_e, 3, 'omitnan');
m_a = mean(a_a, 3, 'omitnan');
