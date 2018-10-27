function [D,y] = get_mamm(varargin)
% Script to download mammographic masses dataset from UCI machine learning repository

% Parse
p = inputParser;
addOptional(p, 'save', false);
addOptional(p, 'impute', false);
parse(p, varargin{:});

%% Start downloading files

fprintf('Starting downloads..')
websave('mammographic.data', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data');
websave('mammographic.names', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.names');

fprintf('Done \n')

%% Call parse script
[D,y] = parse_mamm_gen('save', p.Results.save);

end
