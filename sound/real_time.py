import numpy as np
import sounddevice as sd
import math

# Configuration
FS = 44100  # Sampling rate, Hz
FREQUENCY = 440.0 # Sine frequency, Hz
VOLUME = 0.5 # Range [0.0, 1.0]

def generate_sine_wave(frequency, samplerate, duration):
    """Generates an endless stream of sine wave values."""
    increment = frequency * 2 * math.pi / samplerate
    angle = 0.0
    while True:
        yield VOLUME * math.sin(angle)
        angle += increment
        # Keep angle within 2*pi range to prevent potential float issues over long time
        if angle > 2 * math.pi:
            angle -= 2 * math.pi

def callback(outdata, frames, time, status):
    """Sounddevice callback function to generate and output audio."""
    if status:
        print(status)
    for i in range(frames):
        outdata[i][0] = next(sine_generator)
        # For stereo: outdata[i][1] = next(sine_generator) if needed

# --- Activation Logic ---
print("Activating real-time sine wave audio (press Enter to stop)...")

# Initialize the generator
sine_generator = generate_sine_wave(FREQUENCY, FS, None)

# Open a non-blocking stream
with sd.OutputStream(samplerate=FS, channels=1, callback=callback):
    # The stream runs in the background. The main program waits for user input.
    input()

print("Sound stopped.")
