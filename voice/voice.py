import random

import pydub
import pydub.playback
import pyaudio
import wave
import numpy as np
from scipy import signal

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# Voice discrimination parameters
THRESHOLD = 10  # Adjust based on your environment
NOISE_THRESHOLD = 8000  # Frequency threshold for potential noise
NOISE_BANDS = (0, 150, 2000, RATE // 2)  # Frequency bands to analyze
SILENCE_CHUNKS = 100


def record_audio():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    silent_chunks = 0

    print("Recording...")

    started_recording = False
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)

        # amplitude = np.sqrt(np.mean(np.square(np.frombuffer(data, dtype=np.int16))))
        # print(amplitude)

        frames.append(data)

        # Check for loud noise
        # frequencies, times, spectrogram = signal.spectrogram(data, RATE)
        # noise_bands_energy = np.sum(spectrogram[NOISE_BANDS[1]:NOISE_BANDS[2]], axis=0)
        # if np.max(noise_bands_energy) > NOISE_THRESHOLD:
        #     continue  # Potential noise, skip this chunk

        # if amplitude > THRESHOLD and not started_recording:  # Voice detected
        #     print("Voice detected, starting recording.")
        #     started_recording = True
        #
        # if started_recording:
        #     frames.append(data)
        #
        # if started_recording:
        #     if amplitude > THRESHOLD:  # Voice detected
        #         silent_chunks = 0
        #     else:
        #         silent_chunks += 1
        #         if silent_chunks >= SILENCE_CHUNKS:
        #             print("Silence detected, stopping recording.")
        #             break

    print("Finished recording.")

    # Stop stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames


def play_audio_sped_up(frames):
    rock = pydub.AudioSegment.from_file("rock_template.mp3")
    sound = pydub.AudioSegment(data=b''.join(frames), sample_width=2, frame_rate=int(RATE * 1.5), channels=CHANNELS)

    # Adjust volume
    sound += 40

    print(len(sound))

    silence_length = 3.86 - sound.duration_seconds
    silence = pydub.AudioSegment.silent(duration=silence_length * 1000)
    sound += silence

    rock = rock.overlay(sound, loop=True)
    pydub.playback.play(rock)

    rock.export("rock_saved.wav", format="wav")


if __name__ == "__main__":
    frames = record_audio()
    play_audio_sped_up(frames)
