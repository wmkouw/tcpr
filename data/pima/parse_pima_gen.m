function [D,y] = parse_pima_gen(varargin)
%% Import data from text file.
% Script for importing data from the following text file:
%
%    ./pima-indians-diabetes.data
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2017/05/13 23:19:09

% Parse
p = inputParser;
addOptional(p, 'save', false);
parse(p, varargin{:});

%% Initialize variables.
filename = './pima-indians-diabetes.data';
delimiter = ',';

%% Format for each line of text:

% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this code. If
% an error occurs for a different file, try regenerating the code from the
% Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for unimportable
% data, select unimportable cells in a file and regenerate the script.

%% Allocate imported array to column variable names
D = cell2mat(dataArray(:,1:end-2));
y = cell2mat(dataArray(:,end-1));

y(y==0) = -1;

if p.Results.save
    save('pima', 'D','y');
end 

end
