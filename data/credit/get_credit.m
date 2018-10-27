function [D,y] = get_credit(varargin)
% Script to download credit screening dataset from UCI machine learning repository

% Parse
p = inputParser;
addOptional(p, 'save', false);
addOptional(p, 'impute', false);
parse(p, varargin{:});

%% Start downloading files

fprintf('Starting downloads..')
websave('credit.data', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data');
websave('credit.names', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names');

fprintf('Done \n')

%% Call parse script
[D,y] = parse_credit_gen('save', p.Results.save, 'impute', p.Results.impute);

end




