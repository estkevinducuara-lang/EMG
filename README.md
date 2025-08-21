# EMG
El electromiograma (EMG) es una grabación de la actividad eléctrica de los músculos, también llamada actividad mioeléctrica. Existen dos tipos de EMG, el de superficie y el intramuscular o de aguja. 
Para poder realizar la captura de las señales mioeléctricas se utilizan dos electrodos activos y un electrodo de tierra. En el caso de los electrodos desuperficie, deben ser ubicados en la piel sobre el músculo a estudiar, mientras que el electrodo de tierra se conecta a una parte del cuerpo eléctricamente activa. La señal EMG será la diferencia entre las señales medidas por los electrodos activos. 
Los rangos de frecuencia EMG pueden variar entre 500 Hz y 2500 Hz. 




Para la captura de la señal EMG se utilizó un modulo AD8232 previamente conectado a un sistema de adquisición de datos DAQ, la frecuencia de muestreo que utilizamos fue de 1000 Hz, obteniendo así la siguiente señal EMG.

![saraa](https://github.com/user-attachments/assets/e5db7a64-e507-445b-89e9-930e71df5a64)
*Imagen2. Señal obtenida*

Para poder procesarla se realizo el siguiente codigo:
````
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev4/ai1")
    task.timing.cfg_samp_clk_timing(5000.0, sample_mode=AcquisitionType.FINITE, samps_per_chan=10000)
    data = task.read(READ_ALL_AVAILABLE)
````

## Filtrado de la señal
Se aplicaron dos tipos de filtros, un pasa altas para eliminar los componentes de baja frecuencia y un filtro pasa bajas para elimiinar frecuencias no deseadas, como en nuestro caso ruido ECG; con estos filtros logramos obtener una señal sin ruido como se observa en la imagen. 

![Figure_1](https://github.com/user-attachments/assets/351e1289-3d17-4f86-87eb-072f8036f45d)


*Imagen3. Señal filtrada*

Se obtuvo con el siguiente codigo:
```
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
  ```  

## Aventanamiento 
Para el aventanamiento dividimos la señal usando la ventana Hanning 
Para la comparación de las dos ventanas tanto la de haming como la de hanning, podríamos hablar un poco de cada una para que así mismo podamos definir y decir cual de estas fue la que decidimos utilizar, estas dos son ventanas que suavizan dichas transiciones  	de las señales ayudando así la fuga en la transformada de Fourier. (La fuga para este contexto es el fenómeno ocurrido en el análisis de señales principalmente cuando se realice la transformada de Fourier) 

![image](https://github.com/user-attachments/assets/90528910-0a0b-4e5a-a8e7-574efa362b9f)

*Tabla 1. Diferencia de Hanning y Hamming*

Para este trabajo de laboratorio se decidió trabajar con la ventana de Hanning siendo esta la mejor para la obtención de las señales EEG, también teniendo en cuenta que ofrece una mejor atenuación de la fuga, menos distorsión y reduciendo discontinuidades asegurando así que el análisis espectral sea más preciso obteniendo así estas ventanas de la señal. 
Se obtuvo lo siguiente:

![Imagen de WhatsApp 2025-03-27 a las 21 38 12_5d50efe5](https://github.com/user-attachments/assets/6f34ecea-9ef7-43de-ab3f-e09b7975d086)

*Imagen4. ventanas 1-4*

![Imagen de WhatsApp 2025-03-27 a las 21 39 36_34592a29](https://github.com/user-attachments/assets/10b3d413-5f79-4ec3-8337-22f45ca9c90a)

*Imagen5. ventanas 5-10*

La ventana de Hanning es hacer que cada pedazo empiece y termine suavemente, como si bajaras el volumen poco a poco en los extremos. Así, cuando analizamos la frecuencia de la señal (con la FFT), evitamos que aparezcan ruidos extraños que no deberían estar ahí.

En el código, la ventana de Hanning se aplica multiplicando cada pedazo de la señal por una curva que tiene forma de montaña. Esto ayuda a que el análisis sea mejor.
  ```  
def apply_windowing(signal, window_size, overlap):
    step = int(window_size * (1 - overlap))  
    hanning_window = np.hanning(window_size)  

    return [signal[i:i + window_size] * hanning_window
            for i in range(0, len(signal) - window_size, step)]

window_size = 200  
overlap = 0.3  
windowed_signal = apply_windowing(filtered_signal, window_size, overlap)

  ```  

Obteniendo así de las diferentes ventanas la Transformada de Fourier y el espcetro de frecuencias.

![Imagen de WhatsApp 2025-03-27 a las 21 40 53_3ff49f67](https://github.com/user-attachments/assets/5b7cf6c5-0717-405f-89ed-c8cad7c571af)

*Imagen6. FFT 1-4*

![Imagen de WhatsApp 2025-03-27 a las 21 42 24_be2777b3](https://github.com/user-attachments/assets/10716603-1df6-4cc5-a98e-be44310d1b3a)

*Imagen6. FFT 5-10*

Se realizaron con el siguiente codigo:
  ```  
from scipy.fftpack import fft, fftfreq

def compute_fft(windowed_signal, fs):
    freqs = fftfreq(len(windowed_signal[0]), d=1/fs)[:len(windowed_signal[0]) // 2]
    fft_results = [np.abs(fft(segment))[:len(segment) // 2] for segment in windowed_signal]
    return freqs, fft_results
  ```  

## Análisis Estadístico 
También se obtuvo el análisis estadístico de cada ventana por medio del test de hipótesis, en este caso usamos la hipótesis nula, que dice que la µ10-µ1=0  y haciendo uso de la grafica de dos colas, obteniendo así el valor para t de la ventana 10 y la ventana 1, y se obtuvó el siguiente gráfico de dos colas. 

![grafica dos colas](https://github.com/user-attachments/assets/e7f78ccb-209f-4135-bf1e-01b709df4e9f)

*Imagen24. analisis estadístico*

se obtuvo con el siguiente codigo:
```
x = np.linspace(-4, 4, 1000)
ax.plot(x, t.pdf(x, df=len(windowed_signal[0])-1), label='Distribución t', color='black')
ax.axvline(first_t, color='blue', linestyle='dashed', label=f't primera = {first_t:.2f}')
ax.axvline(last_t, color='red', linestyle='dashed', label=f't última = {last_t:.2f}')
```

### Conclusión
Como se logra observar en el gráfico, el valor de t se encuentra muy cerca al cero, lo que nos indica que no se rechaza la hipótesis nula, no hay evidencia suficiente para decir que la media de la ventana 10 y la ventana 1 sean diferentes, esto nos indica que al momento de la captura EMG el sujeto no llego a la fatiga. 
## Recomendaciones
-Python 3.9, pyedflib, matplotlib, QtWidgets

## Información de contacto
-est.kevin.ducuara@unimilitar.edu.co

## Referencias
Entender la ventana de Hanning: una guía práctica para principiantes. (2024, 14 noviembre). Wray Castle. https://wraycastle.com/es/blogs/knowledge-base/hanning-window?srsltid=AfmBOoqV-xrrZ48jYXADAYPj1efS08lz9JFmbSlEyJHO9bPY93W1iHAp
