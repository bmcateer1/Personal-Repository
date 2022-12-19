function [power, avepow, itc, times] = mttfr(x, fs, freqs, n_cycles,...
    time_bandwidth, useparfor)
% Time-frequency analysis using multitaper wavelets
%
% USAGE:
% ------
%
%   [power, avepow, itc, times] = mttfr(x, fs, freqs, n_cycles, time_bandwidth)
%
% EXAMPLE:
% --------
%
%   [power, avepow, itc, times] = mtffr(x, 4000, 30:5:100, 7.0, 2.0)
%
% x : 2-D array of size n_time x n_trials
%   Data
% fs : scalar, Hz
%   Sampling frequency
% freqs : 1-D vector of size n_freqs x 1
%   The set of frequencies to calculate the time-frequency representation
% n_cycles : either a 1-D vector the same size as freqs (or) scalar
%   The number of cycles in the wavelet to be used
% time_bandwidth : scalar, unitless should be >= 2. Optional, default 4.0
%   Time-(full) bandwith product for the wavelet tapers. The number of
%   tapers is automatically chosen based on the time-bandwidth product.
% useparfor : boolean, optional, default is false
%   Set to true if parfol is available for your MATLAB version and machine
%   setup to possibly reduce computation time by using parallel threads for
%   different frequencies. Not recommended for small datasets.
%
% References:
% -----------
% Slepian, D. (1978). Prolate spheroidal wave functions, Fourier analysis,
% and uncertainty?V: The discrete case. Bell System Technical Journal,
% 57(5), 1371-1430.
%
% Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
% Proceedings of the IEEE, 70(9), 1055-1096.
%
% ------------------------------------------------------------------------
% Copyright 2015-2019 Hari Bharadwaj. All rights reserved.
% Hari Bharadwaj <hari.bharadwaj@gmail.com>
% ------------------------------------------------------------------------

if ~ismatrix(x)
    error('Data should be a 2D array');
end

[n_times, n_trials] = size(x);
times = (0:(n_times-1))/fs;

if ~isvector(freqs)
    error('Set of frequencies should be a vector');
end

n_freqs = numel(freqs);

if ~isscalar(n_cycles) && (numel(n_cycles) ~= n_freqs)
    error('n_cycles should be a scalar or the same size as freqs')
end

if isscalar(n_cycles)
    n_cycles = n_cycles * ones(size(freqs));
end

if ~exist('time_bandwidth', 'var')
    time_bandwidth = 4.0;
end

n_taps = floor(time_bandwidth - 1);

if ~(n_taps > 0)
    error('Time-bandwidth product should be at least 2');
end

power = zeros(n_freqs, n_times);
itc = zeros(n_freqs, n_times);
avepow = zeros(n_freqs, n_times);

if ~exist('useparfor', 'var')
    useparfor = 0;
end

if(useparfor)
    fprintf(1, 'Performing parallel decomposition for %d frequencies\n',...
        n_freqs);
    parfor k_freq = 1:n_freqs
        n_cycles_k = n_cycles(k_freq);
        freq_k = freqs(k_freq);
        n_samps = round(fs * n_cycles_k / freq_k);
        t_k = (0:(n_samps - 1))/fs;
        t_k = t_k - mean(t_k);  % Center around time 0
        tap = dpss(n_samps, time_bandwidth/2.0, n_taps);
        
        tfr = zeros(n_taps, n_times, n_trials);
        for k_tap = 1:n_taps
            w = exp(1j*2*pi*freq_k*t_k) .* ...
                (tap(:, k_tap)' - 0.5*tap(1, k_tap)' - 0.5*tap(end, k_tap)');
            w = w ./ sum(w);
            tfr(k_tap, :, :) = conv2(x, w(:), 'same');
        end
        power(k_freq, :) = squeeze(mean(mean(abs(tfr), 3), 1));
        avepow(k_freq, :) = abs(squeeze(mean(mean(tfr, 3), 1)));
        itc(k_freq, :) = abs(squeeze(mean(mean(tfr, 3), 1))) ./...
            power(k_freq, :);
    end
    
else
    fprintf(1, 'Performing decomposition serially for %d frequencies\n',...
        n_freqs);
    for k_freq = 1:n_freqs
        n_cycles_k = n_cycles(k_freq);
        freq_k = freqs(k_freq);
        n_samps = round(fs * n_cycles_k / freq_k);
        t_k = (0:(n_samps - 1))/fs;
        t_k = t_k - mean(t_k);  % Center around time 0
        tap = dpss(n_samps, time_bandwidth/2.0, n_taps);
        
        tfr = zeros(n_taps, n_times, n_trials);
        for k_tap = 1:n_taps
            w = exp(1j*2*pi*freq_k*t_k) .* ...
               (tap(:, k_tap)' - 0.5*tap(1, k_tap)' - 0.5*tap(end, k_tap)');
            w = w ./ norm(w);
            tfr(k_tap, :, :) = conv2(x, w(:), 'same');
        end
        power(k_freq, :) = squeeze(mean(mean(abs(tfr).^2, 3), 1));
        avepow(k_freq, :) = abs(squeeze(mean(mean(tfr, 3).^2, 1)));
        itc(k_freq, :) = abs(squeeze(mean(mean(tfr, 3).^2, 1))) ./ ...
            power(k_freq, :);
        fprintf(1,...
            'Calculated coefficients for %d / %d frequencies...\n',...
            k_freq, n_freqs);
    end
end