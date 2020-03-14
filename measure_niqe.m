clear variables; clc;

%%
% folder = 'datas/KCMI-550'; ext = '*.jpg';
folder = 'results/KCMI-550-Ours'; ext = '*.png';
% folder = 'datas/VISION-1500'; ext = '*.jpg';
% folder = 'results/VISION-1500-Ours'; ext = '*.png';
% folder = 'results/KCMI-550-Crop'; ext = '*.png';
% folder = 'results/KCMI-550-Crop-Ours'; ext = '*.png';
filepaths = dir(fullfile(folder, ext));

scores = zeros(length(filepaths), 1);
parfor i = 1 : length(filepaths)
	im = imread(fullfile(folder, filepaths(i).name));
	score = niqe(im); % tested on MATLAB 2018b
	scores(i) = score;
	disp({i, score});
end
disp(mean(scores));
