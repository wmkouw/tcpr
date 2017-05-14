function [D,y] = get_bands(varargin)
% Script to download cylinder bands dataset from UCI machine learning repository

% Parse
p = inputParser;
addOptional(p, 'save', false);
addOptional(p, 'impute', true);
parse(p, varargin{:});

%% Start downloading files

fprintf('Starting downloads..')
websave('bands.data', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/cylinder-bands/bands.data');
websave('bands.names', ...
    'https://archive.ics.uci.edu/ml/machine-learning-databases/cylinder-bands/bands.names');

fprintf('Done \n')

% Warning
sprintf('There is a problem with the bands.data file from UCI repository:\nat line 522 a line break has been inserted that should not have been there.\n \nPlease remove the break (script paused until complete) \n')
pause

%% Call parse script
[D,y] = parse_bands_gen('save', p.Results.save);

end
