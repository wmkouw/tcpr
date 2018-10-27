% Script to gather results from domain adaptation experiments on cylinder bands
% dataset

close all;
clearvars;

% Load data
dataName = 'bands';
load(dataName)
saveName = 'results/';

% Experimental parameters
clfs = {'tca', 'iwc_kmm_lsq', 'rcsa', 'rba', 'tcp-ls', 'tcp-lda', 'tcp-qda'};
clfs_names = {'tca', 'kmm', 'rcsa', 'rba', 'tcp_ls', 'tcp_lda', 'tcp_qda'};
nR = 1;
no = '1';
prep = 'minusminmaxdivimpute0';

% Number of classifiers and combinations
nCl = length(clfs);

% Preallocate
risks = NaN(nCl,nR);
error = NaN(nCl,nR);
areas = NaN(nCl,nR);
for c = 1:length(clfs)
    
    clear R e AUC
    switch clfs{c}
        case {'sls','tls'}
            load([saveName dataName '_prep' prep '_nR' num2str(nR) '_results_ssb_tcp-ls_' no '.mat']);
        case {'slda','tlda'}
            load([saveName dataName '_prep' prep '_nR' num2str(nR) '_results_ssb_tcp-lda_'  no '.mat']);
        case {'sqda','tqda'}
            load([saveName dataName '_prep' prep '_nR' num2str(nR) '_results_ssb_tcp-qda_'  no '.mat']);
        otherwise
            load([saveName dataName '_prep' prep '_nR' num2str(nR) '_results_ssb_' clfs{c} '_' no '.mat']);
    end
    
    for r = 1:nR
        
        switch clfs{c}
            case {'sls', 'slda', 'sqda'}
                % Risk
                risks(c,r) = R{r}.ref_u;
                
                % Error rate
                error(c,r) = e{r}.ref_u;
                
                % AUC
                areas(c,r) = AUC{r}.ref_u;
            case {'tcp-ls', 'tcp-lda', 'tcp-qda'}
                % Risk
                risks(c,r) = R{r}.tcp_u;
                
                % Error rate
                error(c,r) = e{r}.tcp_u;
                
                % AUC
                areas(c,r) = AUC{r}.tcp_u;
            case {'tls', 'tdla', 'tqda'}
                % Risk
                risks(c,r) = R{r}.orc_u;
                
                % Error rate
                error(c,r) = e{r}.orc_u;
                
                % AUC
                areas(c,r) = AUC{r}.orc_u;
            otherwise
                % Risk
                risks(c,r) = R(r);
                
                % Error rate
                error(c,r) = e(r);
                
                % AUC
                areas(c,r) = AUC(r);
        end
    end
end

% Average over repeats
mean_risks = mean(risks, 3, 'omitnan');
mean_error = mean(error, 3, 'omitnan');
mean_areas = mean(areas, 3, 'omitnan');

% Combine into table
T = array2table(mean_areas', 'VariableNames', clfs_names)

