clear variables; clc;
addpath('libs/CameraFingerprint');
addpath('libs/CameraFingerprint/Filter');
addpath('libs/CameraFingerprint/Functions');

%%
folder = 'results/KCMI-550-Crop-Ours';
[name, ~] = textread('datas/KCMI-550-Val', '%s %d');
for i = 1 : length(name)
	name{i} = [name{i}(1:end-3),'png'];
end

%%
record = zeros(10,1);
for i = 1 : 10
 	% load fingerprint for a camera model
	F = load(sprintf('results/KCMI-Fingerprint/%d',i-1));
	Fingerprint = F.Fingerprint;
	names = name(55*(i-1)+1:55*i);
	
	% Peak-to-Correlation Energy ratio
	pces = zeros(length(names),1);
	parfor j = 1 : length(names)
		imx = fullfile(folder, names{j});
		Noisex = NoiseExtractFromImage(imx, 2);
		Noisex = WienerInDFT(Noisex, std2(Noisex));
		
		Ix = double(rgb2gray(imread(imx)));
		C = crosscorr(Noisex, Ix.*Fingerprint);
		[~, pce0] = PCE(C);
		pces(j) = pce0.PCE;
	end
	disp({i, mean(pces)});
	record(i,1) = mean(pces);
end
fprintf('Avg: %.4f\n', mean(record(:,1)));
