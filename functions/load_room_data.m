function [H, r, fVec] = load_room_data()
% Load data for empty_cuboid_room dataset. It requirst to first download
% the data set from https://doi.org/10.11583/DTU.13315289.v1
%
% The function removes measured postions that are duplicated and sorts the 
% positions. A time window is aplied to the IRs to reduce RT.
% Outputs:
%   H: frequency response
%   r: positions
%   fVec: vector of frequency bins

% h5info('empty_cuboid_room_frequency_responses.h5')
% h5disp('empty_cuboid_room_frequency_responses.h5')

fVec = h5read('./data/empty_cuboid_room_frequency_responses.h5',...
    '/position_0/frequency');
HAll = [];
rAll = [];

for i = 1:6
    H = h5read('./data/empty_cuboid_room_frequency_responses.h5',...
        ['/position_' num2str(i) '/response']);
    H = H.r + 1i*H.i;
    HAll = [HAll; H];
    
    r = h5read('./data/empty_cuboid_room_frequency_responses.h5',...
        ['/position_' num2str(i) '/xyz']);
    r = r.';
    rAll = [rAll; r];
end
r = rAll/1000;
r = [r(:,2) r(:,1)];
r = r-[2.2 1.65]; % centering
r = round(r,2);
r = [r, zeros(size(r,1),1)];

H = HAll;
clear rAll HAll

% Find positions that were measured twice
[rUnique, ia, ib] = unique(round(r,10), 'rows', 'stable');
iDupRows = setdiff(1:size(r,1), ia);
rDuplicate = r(iDupRows,:);

r = rUnique;
H = H(ia,:);
clear ia ib iDupRows rDuplicate rUnique i

% Sort positions
[r,iSort] = sortrows(r);
H = H(iSort,:);

% 1 Hz resolution
fVec = fVec(1:10:end);
H = H(:,1:10:end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get IR
H2 = [H fliplr(conj(H(:,2:end)))];
h = ifft(H2,[],2);

% Apply time window 
fs = fVec(end)*2;
nWin = 0.1*fs*2; % 0.1 s window
winTuk = tukeywin(nWin,1);
win = zeros(1,size(h,2));
win(1:nWin/2) = winTuk(nWin/2+1:end);
h = h.*win;

H2 = fft(h,[],2);
H = H2(:,1:(end+1)/2);
H = H/max(abs(H(:)));

% Normalize
H = H/max(abs(H(:)));

end