import numpy as np
import sounddevice as sd

def white_noise(duration: float=1.0, amplitude: float=0.5, sample_rate: int=44100) -> np.ndarray:
    n_samples = int(duration * sample_rate)
    noise = np.random.uniform(-1, 1, n_samples)
    noise *= amplitude
    return noise

def main():
    mysound = white_noise()
    sd.play(mysound)
    sd.wait()

if __name__ == "__main__":
    main()