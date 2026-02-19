import numpy as np
import sounddevice as sd

def main():
    mysound = sine_wave(duration=0.1)
    sd.play(mysound)
    sd.wait()

def white_noise(duration: float=1.0, amplitude: float=0.5, sample_rate: int=44100) -> np.ndarray:
    n_samples = int(duration * sample_rate)
    noise = np.random.uniform(-1, 1, n_samples)
    noise *= amplitude
    return noise

def sine_wave(duration: float=1.0, amplitude: float=0.5, sample_rate: int=44100, frequency: float=440) -> np.ndarray:
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, False) # generate an array of evenly spaced time points
    sine = np.sin(2 * np.pi * frequency * t)
    sine *= amplitude
    return sine

if __name__ == "__main__":
    main()