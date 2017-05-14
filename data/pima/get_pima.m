function [D,y] = get_pima(varargin)
% Script to download pima diabetes dataset from UCI machine learning repository

% Parse
p = inputParser;
addOptional(p, 'save', false);
parse(p, varargin{:});

%% Start downloading files

fprintf('Starting downloads..')
websave('pima-indians-diabetes.data', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data');
websave('pima-indians-diabetes.names', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names');

fprintf('Done \n')

%% Call parse script
[D,y] = parse_pima_gen('save', p.Results.save);

end
