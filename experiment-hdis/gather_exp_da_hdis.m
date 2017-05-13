% Script to gather results from domain adaptation experiments on heart disease

close all;
clearvars;

% Load data
dataname = 'hdis_imp0';
load(dataname)

% Experimental parameters
clfs = {'tca', 'kmm', 'rcsa', 'rba', 'tcp-ls', 'tcp-lda', 'tcp-qda'};
cc = 1:12;
nR = 10;
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
                case {'sls', 'tcp-ls', 'tls'}
                    % Risk
                    a_r(c,n,r) = R{r}.ref_u;
                    
                    % Error rate
                    a_e(c,n,r) = e{r}.ref_u;
                    
                    % AUC
                    a_a(c,n,r) = AUC{r}.ref_u;
                case {'tcp-ls', 'tcp-lda', 'tcp-qda'}
                    % Risk
                    a_r(c,n,r) = R{r}.tcp_u;
                    
                    % Error rate
                    a_e(c,n,r) = e{r}.tcp_u;
                    
                    % AUC
                    a_a(c,n,r) = AUC{r}.tcp_u;
                case {'tls', 'tdla', 'tqda'}
                    % Risk
                    a_r(c,n,r) = R{r}.orc_u;
                    
                    % Error rate
                    a_e(c,n,r) = e{r}.orc_u;
                    
                    % AUC
                    a_a(c,n,r) = AUC{r}.orc_u;
                otherwise
                    % Risk
                    a_r(c,n,r) = R(r);
                    
                    % Error rate
                    a_e(c,n,r) = e(r);
                    
                    % AUC
                    a_a(c,n,r) = AUC(r);
            end
        end
    end
end

% Average over repeats
m_r = mean(a_r, 3, 'omitnan');
m_e = mean(a_e, 3, 'omitnan');
m_a = mean(a_a, 3, 'omitnan');

% Combine into table
clfs_names = {'tca', 'kmm', 'rcsa', 'rba', 'tcp_ls', 'tcp_lda', 'tcp_qda'};
T = array2table(m_a', 'VariableNames', clfs_names)
