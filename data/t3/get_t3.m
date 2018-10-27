function [D,y] = get_t3(varargin)
% Script to download tic-tac-toe dataset from UCI machine learning repository

% Parse
p = inputParser;
addOptional(p, 'save', false);
parse(p, varargin{:});

%% Start downloading files

fprintf('Starting downloads..')
websave('tic-tac-toe.data', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data');
websave('tic-tac-toe.names', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.names');

fprintf('Done \n')

%% Call parse script
[D,y] = parse_t3_gen('save', p.Results.save);

end
