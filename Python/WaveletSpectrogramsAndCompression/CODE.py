# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:44:04 2021

@author: Ben
"""

#PS4
import numpy as np
import matplotlib.pyplot as plt




#3A
fs = 10000# sampling rate given
std = 0.5 #given standard deviation
noisemean = 0
t = np.arange(0,1,1/fs)


phi = []
signalf = []

phi =  2 * np.pi * (100*t + 400/3*(t**3))
signalf= np.cos(phi)  #Chirp

noise = np.random.normal(noisemean,std, len(t))

messychirp = signalf + noise

plt.figure()
plt.plot(t,messychirp)
plt.title('3A: One second Chirp with white noise')
plt.xlabel('Time (S)')
plt.ylabel('Amplitude')


#3B
from numpy.fft import fft, ifft
from scipy import signal

signalFFT = np.fft.fft(messychirp,fs)
signalfreq = np.arange(0,fs)

#signalfreq = np.fft.fftfreq(len(signalFFT))

signalFFT = signalFFT[signalfreq < fs/2]
signalfreq = signalfreq[signalfreq < fs/2]

signalperiod = (np.abs(signalFFT)**2) / fs #may need to change Order of Operations
signalperiod /= signalperiod.sum()


N = fs # 1 second duration
NW = 4
Kmax = 4

wins, concentrations = signal.windows.dpss(N, NW, Kmax=Kmax, return_ratios=True)

periodfreq = []
specsignal = []

for i in range(0,3):
    currentfreq, currSignal = signal.periodogram(messychirp, window=wins[i], fs=fs)
    periodfreq.append(currentfreq)
    specsignal.append(currSignal)
    
specsignal = np.mean(specsignal,axis = 0)

plt.figure()
plt.plot(signalfreq, 10*np.log10(signalperiod), label = "Signal")
plt.plot(periodfreq[0], 10*np.log10(specsignal), label = "Estimate")
plt.title('3B: Frequency Spectrum Estimate Compared with Multitaper Estimate')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0,1000)
plt.legend()

# plt.figure()
# for i in range(0,4):
#     plt.plot(t,wins[i])
# plt.title('First 4 DPSS Tapers')
# plt.xlabel('Time (S)')
# plt.ylabel('Amplitude')

##3C
revphi = 2 * np.pi * (100*t + 400/3*(-t**3))
revsignal = np.cos(phi)
revmessychirp = revsignal + noise

revsignalFFT = np.fft.fft(revmessychirp,fs)
revsignalfreq = np.arange(0,fs)

#signalfreq = np.fft.fftfreq(len(signalFFT))

revsignalFFT = revsignalFFT[revsignalfreq < fs/2]
revsignalfreq = revsignalfreq[revsignalfreq < fs/2]

revsignalperiod = (np.abs(revsignalFFT)**2) / fs #may need to change Order of Operations
revsignalperiod /= revsignalperiod.sum()

revwins, revconcentrations = signal.windows.dpss(N, NW, Kmax=Kmax, return_ratios=True)

revperiodfreq = []
revspecsignal = []

for i in range(0,3):
    revcurrentfreq, revcurrSignal = signal.periodogram(revmessychirp, window=revwins[i], fs=fs)
    revperiodfreq.append(revcurrentfreq)
    revspecsignal.append(revcurrSignal)
    
revspecsignal = np.mean(revspecsignal,axis = 0)

plt.figure()
plt.plot(revsignalfreq, 10*np.log10(revsignalperiod), label = "Reverse Signal")
plt.plot(revperiodfreq[0], 10*np.log10(revspecsignal), label = "Reverse Estimate")
plt.title('3C: Reverse Signal Spectrum Compared with Multitaper Estimate')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0,1000)
plt.legend()

#3D
import tfr

freqs = np.arange(50,800,10) #from his code- lower freq, higher freq, length (in Hz)
n_cycles = 7
time_bandwidth = 2
psd, t = tfr.tfr_multitaper(messychirp[None, None, :], fs, frequencies=freqs,
                            time_bandwidth=2, n_cycles=n_cycles)

plt.figure()
plt.pcolormesh(t, (freqs), 10 * np.log10(psd.squeeze()), cmap='RdYlBu_r', shading='auto')
plt.title('Chirp Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')


#rev chirp spectrogram
revpsd, revt = tfr.tfr_multitaper(revmessychirp[None, None, :], fs, frequencies=freqs,
                            time_bandwidth=2, n_cycles=n_cycles)

plt.figure()
plt.pcolormesh(-revt,(freqs), 10 * np.log10(revpsd.squeeze()), cmap='RdYlBu_r', shading='auto')
plt.title('Reversed Chirp Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

#3E
time_bandwidth=5 #number of tapers +1
psd4, t4 = tfr.tfr_multitaper(messychirp[None, None, :], fs, frequencies=freqs,
                            time_bandwidth=time_bandwidth, n_cycles=n_cycles)

plt.figure()
plt.pcolormesh(t4, (freqs), 10 * np.log10(psd4.squeeze()), cmap='RdYlBu_r', shading='auto')
plt.title('Chirp Spectrogram 4 tapers')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')


#rev chirp spectrogram
revpsd4, revt4 = tfr.tfr_multitaper(revmessychirp[None, None, :], fs, frequencies=freqs,
                            time_bandwidth=time_bandwidth, n_cycles=n_cycles)

plt.figure()
plt.pcolormesh(-revt4, (freqs), 10 * np.log10(revpsd4.squeeze()), cmap='RdYlBu_r', shading='auto')
plt.title('Reversed Chirp Spectrogram 4 tapers')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')




#4a
from IPython.display import Audio
from scipy.io import loadmat

speechdata = loadmat('speech', squeeze_me = True)

ba = speechdata['ba']
da = speechdata['da']
fs = speechdata['fs']

#time = np.arange(0, ba.shape[0]) / fs

freqs = np.arange(32,2048,20)#not confident that the bins are correct. Need 20 per octive
n_cycles = 9

bapsd, bat = tfr.tfr_multitaper(ba[None, None, :], fs, frequencies=freqs,
                             time_bandwidth=2, n_cycles=n_cycles)

plt.figure()
plt.pcolormesh(bat, freqs, bapsd.squeeze(), cmap='RdYlBu_r', shading='auto')
plt.title('4A: "Ba" Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

dapsd, dat = tfr.tfr_multitaper(da[None, None, :], fs, frequencies=freqs,
                             time_bandwidth=2, n_cycles=n_cycles)

plt.figure()
plt.pcolormesh(dat, freqs, dapsd.squeeze(), cmap='RdYlBu_r', shading='auto')
plt.title('4A: "Da" Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

#Audio(data=ba, rate =fs) #plays unfiltered audio


#4b


##5a
import pywt
#print(pywt.families(short=False))
tba = np.arange(0, ba.shape[0]+1) / fs
tda = np.arange(0, da.shape[0]+1) / fs
coeffs_ba = pywt.wavedec(ba,wavelet='coif3', level=5)
coeffs_da = pywt.wavedec(da,wavelet='coif3', level=5)

# Convert to array -- easier for thresholding
baarr, baslices = pywt.coeffs_to_array(coeffs_ba)
bathresh = np.percentile(abs(baarr),66) #np.percentile finds the requested percentileof data via magnitude
baarr[abs(baarr) < bathresh] = 0
baarr[baarr > bathresh] -= bathresh
baarr[baarr < -bathresh] += bathresh


daarr, daslices = pywt.coeffs_to_array(coeffs_da)
dathresh = np.percentile(abs(daarr),66)
daarr[abs(daarr) < dathresh] = 0
daarr[daarr > dathresh] -= dathresh
daarr[daarr < -dathresh] += dathresh

# Convert back to wavedec/waverec format
bacoeffs_denoised = pywt.array_to_coeffs(baarr, baslices, output_format='wavedec')
barecon_denoised = pywt.waverec(bacoeffs_denoised, wavelet='coif3')
plt.figure()
plt.plot(tba, barecon_denoised)

Audio(data=barecon_denoised, rate =fs)

dacoeffs_denoised = pywt.array_to_coeffs(daarr, daslices, output_format='wavedec')
darecon_denoised = pywt.waverec(dacoeffs_denoised, wavelet='coif3')
plt.figure()
plt.plot(tda, darecon_denoised)

Audio(data=darecon_denoised, rate =fs)
#5b
#80%

##SWITCHED TO JUPYTER
chestradiograph = loadmat('chestradiograph', squeeze_me = True)


















