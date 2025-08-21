import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.fftpack import fft, fftfreq
from scipy.stats import ttest_1samp, t
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ------------------ Funciones auxiliares ------------------

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en {image_path}")
    return image

def extract_signal(image):
    return np.mean(image, axis=1)

def apply_filter(signal, cutoff, fs, filter_type='high'):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    sos = butter(4, normalized_cutoff, btype=filter_type, analog=False, output='sos')
    return sosfiltfilt(sos, signal)

def normalize_to_voltage(signal, v_max=5.0):
    signal = signal - np.min(signal)
    return (signal / np.max(signal)) * v_max

def apply_windowing(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))
    return [signal[i:i + window_size] for i in range(0, len(signal) - window_size, step)]

def compute_fft(windowed_signal, fs):
    freqs = fftfreq(len(windowed_signal[0]), d=1/fs)[:len(windowed_signal[0]) // 2]
    fft_results = [np.abs(fft(segment))[:len(segment) // 2] for segment in windowed_signal]
    return freqs, fft_results

# ------------------ Clases para interfaz gráfica ------------------

class SignalWindow(QWidget):
    def __init__(self, window_index, segment):
        super().__init__()
        self.setWindowTitle(f'Señal - Ventana {window_index + 1}')
        layout = QVBoxLayout()
        fig, ax = plt.subplots()
        time_axis = np.linspace(0, len(segment) / 1000, len(segment))
        ax.plot(time_axis, segment, label=f'Ventana {window_index + 1}', color='blue')
        ax.set_title('Señal en el Tiempo')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Voltaje (V)')
        ax.legend()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)

class FFTWindow(QWidget):
    def __init__(self, window_index, fft_values, freqs):
        super().__init__()
        self.setWindowTitle(f'FFT - Ventana {window_index + 1}')
        layout = QVBoxLayout()
        fig, ax = plt.subplots()
        ax.plot(freqs, fft_values, label=f'FFT Ventana {window_index + 1}', color='red')
        ax.set_xlim([0, 500])
        ax.set_title('FFT de la Ventana')
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('Amplitud')
        ax.legend()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)

# ------------------ Función principal ------------------

def main():
    image_path = 'saraa.png'
    
    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen en {image_path}")
        sys.exit(1)
    
    image = load_image(image_path)
    signal = extract_signal(image)
    
    fs = 1000
    high_cutoff = 20
    low_cutoff = 450
    
    filtered_high = apply_filter(signal, high_cutoff, fs, filter_type='high')
    filtered_signal = apply_filter(filtered_high, low_cutoff, fs, filter_type='low')
    filtered_signal = normalize_to_voltage(filtered_signal, v_max=5.0)
    full_mean = np.mean(filtered_signal)

    window_size = 200
    overlap = 0.3
    windowed_signal = apply_windowing(filtered_signal, window_size, overlap)
    freqs, fft_results = compute_fft(windowed_signal, fs)
    
    first_t, last_t = None, None
    
    app = QApplication(sys.argv)
    windows_list = []
    
    for i, (segment, fft_values) in enumerate(zip(windowed_signal, fft_results)):
        if len(segment) > 1:
            mean_val = np.mean(segment)
            t_stat, _ = ttest_1samp(segment, full_mean)
            if i == 0:
                first_t = t_stat
            if i == len(windowed_signal) - 1:
                last_t = t_stat
        
        signal_window = SignalWindow(i, segment)
        fft_window = FFTWindow(i, fft_values, freqs)
        
        signal_window.show()
        fft_window.show()
        windows_list.extend([signal_window, fft_window])
    
    if first_t is not None and last_t is not None:
        fig, ax = plt.subplots()
        x = np.linspace(-4, 4, 1000)
        ax.plot(x, t.pdf(x, df=len(windowed_signal[0])-1), label='Distribución t', color='black')
        ax.axvline(first_t, color='blue', linestyle='dashed', label=f't primera = {first_t:.2f}')
        ax.axvline(last_t, color='red', linestyle='dashed', label=f't última = {last_t:.2f}')
        ax.set_title('Distribución t de Student')
        ax.set_xlabel('t')
        ax.set_ylabel('Probabilidad')
        ax.legend()
        plt.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
