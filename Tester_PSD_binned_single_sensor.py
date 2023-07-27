import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import cumtrapz

# Load raw vibration data
file='RT10min.csv'
skip_rows = 24

data = np.loadtxt(file, delimiter=',', skiprows=skip_rows, usecols= (3,5,7))
data=data/1000

timestep = np.loadtxt(file, delimiter=',', skiprows=skip_rows, usecols= (0))

# Calculate the sampling frequency
fs = 200

#Calculate the integrals of data to identify any visual trends
vel_data=cumtrapz(timestep, data[:,2], initial=0)
pos_data=cumtrapz(timestep, vel_data, initial=0)

# Apply FFT to the data
fft_data = fft(data)

# Compute the frequencies corresponding to FFT coefficients
freqs = fftfreq(len(data), 1/fs)

# Ignoring the Negative values
mask= freqs>0

# Calculate the power spectrum
psd = np.abs(fft_data) ** 2

# Define window size and overlap
window_size = 2048
overlap = 0.5  # 50% overlap

# Calculate the number of overlapping samples
overlap_samples = int(window_size * overlap)

# Calculate the number of windows
num_windows = int((len(data) - overlap_samples) / (window_size - overlap_samples))

# Initialize arrays for the binned PSD and frequency bins
binned_psd = np.zeros((window_size // 2, data.shape[1]))
frequency_bins = np.arange(0, fs/2, fs/window_size)

# Apply Hanning window and calculate binned PSD for each window and each column
for i in range(num_windows):
    start = i * (window_size - overlap_samples)
    end = start + window_size
    windowed_data = data[start:end, :] * np.hanning(window_size)[...,np.newaxis]
    windowed_fft = fft(windowed_data, axis=0)
    windowed_power_spectrum = np.abs(windowed_fft) ** 2
    binned_psd += windowed_power_spectrum[:window_size // 2, :]

# Normalize the binned PSD
binned_psd /= num_windows

# Plot each column in separate subplots with shared x and y axes

fig, axs = plt.subplots(data.shape[1], 1, sharex=True, sharey=True, figsize=(8, 10))
fig.suptitle('Binned PSD for Bin size:'+str(window_size)+' timesteps @ 200Hz sampling rate')
for col in range(data.shape[1]):
    axs[col].plot(frequency_bins, binned_psd[:, col])
#axs[0].set_ylabel('Z-axis 3602/ A^2/Hz')
axs[0].set_ylabel('Z-axis 3604 Ch3/ A^2/Hz')
axs[1].set_ylabel('X-axis 3604 Ch1/ A^2/Hz')
axs[2].set_ylabel('Y-axis 3604 Ch2/ A^2/Hz')

axs[-1].set_xlabel('Freq. / Hz)')

plt.tight_layout()

# Plot the PSD

fig2=plt.figure(2)
fig2.suptitle('Power Spectral Density vs Frequency @ 200 Hz sampling rate')

ax1= plt.subplot(311)
plt.plot(freqs[mask], psd[mask,0])
#plt.ylabel('Z-axis 3602/ A^2/Hz')

ax2= plt.subplot(312, sharex=ax1, sharey=ax1)
plt.plot(freqs[mask], psd[mask,1])
#plt.ylabel('Z-axis 3604 Ch3/ A^2/Hz')

ax3= plt.subplot(313, sharex=ax1, sharey=ax1)
plt.plot(freqs[mask], psd[mask,2])
#plt.ylabel('X-axis 3604 Ch1/ A^2/Hz')

#ax4= plt.subplot(414, sharex=ax1, sharey=ax1)
#plt.plot(freqs[mask], psd[mask,3])
#plt.ylabel('Y-axis 3604 Ch1/ A^2/Hz')
plt.xlabel('Freq. / Hz')

#Plot the raw, integral and double integral data to identify any visual trends

fig3=plt.figure(3)
fig3.suptitle('Plotting of Data with Successive Integrals to Visually identify any Trends @ 200Hz sampling rate')

axis1= plt.subplot(311)
plt.plot(timestep, data[:,1])
plt.ylabel('Acc. Data / m/s^2')

axis2= plt.subplot(312, sharex=axis1)
plt.plot(timestep, vel_data)
plt.ylabel('Vel. Data / m/s')

axis2= plt.subplot(313, sharex=axis1)
plt.plot(timestep, pos_data)
plt.ylabel('Pos. Data / m')
plt.xlabel('Time / ms')

plt.show()

