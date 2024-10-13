
import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load the audio file
file_path = 'your_audio_file.wav'
y, sr = librosa.load(file_path, sr=None)

# Apply low-pass filter
def low_pass_filter(y, sr, cutoff=1000, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y_filtered = signal.filtfilt(b, a, y)
    return y_filtered

y_filtered = low_pass_filter(y, sr)

# Perform spectral subtraction
def spectral_subtract(y, sr, noise_factor=0.5):
    spec = librosa.stft(y)
    mag, phase = librosa.magphase(spec)
    noise_mag = np.mean(mag[:, :5], axis=1)
    mag_sub = mag - noise_factor * noise_mag[:, np.newaxis]
    mag_sub = np.maximum(mag_sub, 0)
    y_denoised = librosa.istft(mag_sub * phase)
    return y_denoised

y_denoised = spectral_subtract(y_filtered, sr)

# Save and display the results
librosa.output.write_wav('denoised_audio.wav', y_denoised, sr)

# Plot the waveforms
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Original')
plt.subplot(3, 1, 2)
librosa.display.waveshow(y_filtered, sr=sr)
plt.title('Low-pass Filtered')
plt.subplot(3, 1, 3)
librosa.display.waveshow(y_denoised, sr=sr)
plt.title('Denoised')
plt.tight_layout()
plt.show()
