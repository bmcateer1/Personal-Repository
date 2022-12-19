function plottfr(t, f, S, fscale)
% Just a simple function to plot time-frequency images. Does not do any
% checks for compatibility of the three inputs.
% USAGE: plottfr(t, f, S)
%
% INPUTS:
%   t - time vector (in seconds of length n_times say)
%   f - frequency vector (in Hz of length n_freqs say)
%   S - timefrequency spectrogram (size n_freqs x n_times)
%   fscale - whether to plot f in 'linear'[default] or 'log' scale (this is
%            regardless of whether the spacing of f is log or linear)
%
% ------------------------------------------------------------------------
% Copyright 2015-2019 Hari Bharadwaj. All rights reserved.
% Hari Bharadwaj <hari.bharadwaj@gmail.com>
% ------------------------------------------------------------------------

if ~exist('fscale', 'var')
    fscale = 'linear';
end

figure;    
h = pcolor(t, f, S);
h.EdgeColor = 'none';
shading('interp');
xlabel('Time (s)', 'FontSize', 16);
ylabel('Frequency (Hz)', 'FontSize', 16);
axis xy;
ax = gca;
set(ax, 'FontSize', 16, 'yscale', fscale);
ax.YAxis.Exponent = 0;