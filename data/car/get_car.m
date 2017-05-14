function [D,y] = get_car(varargin)
% Script to download car evaluation dataset from UCI machine learning repository

% Parse
p = inputParser;
addOptional(p, 'save', false);
parse(p, varargin{:});

%% Start downloading files

fprintf('Starting downloads..')
websave('car.data', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data');
websave('car.names', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.names');

fprintf('Done \n')

%% Call parse script
[D,y] = parse_car_gen('save', p.Results.save);

end
