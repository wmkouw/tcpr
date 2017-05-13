function [D,y] = get_iono(varargin)
% Script to download ionosphere dataset from UCI machine learning repository

% Parse
p = inputParser;
addOptional(p, 'save', false);
parse(p, varargin{:});

%% Start downloading files

fprintf('Starting downloads..')
websave('ionosphere.data', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data');
websave('ionosphere.names', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.names');

fprintf('Done \n')

%% Call parse script
[D,y] = parse_iono_gen('save', p.Results.save);

end
