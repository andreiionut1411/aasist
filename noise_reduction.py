import os
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from pydub import AudioSegment
from p_tqdm import p_map

def highpass_filter(data, sr, cutoff=100):
    """
    Apply a high-pass filter to remove low-frequency noise.

    Args:
        data (numpy.array): Audio signal.
        sr (int): Sample rate.
        cutoff (int): Cutoff frequency (default 100Hz).

    Returns:
        numpy.array: Filtered audio.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def process_audio(input_file, output_file):
    """
    Process an audio file by applying noise reduction, high-pass filtering, and volume normalization.

    Args:
        input_file (str): Path to the input WAV file.
        output_file (str): Path to save the cleaned WAV file.
    """
    try:
        # Load audio
        y, sr = librosa.load(input_file, sr=None)

        # Noise reduction
        y_denoised = nr.reduce_noise(y=y, sr=sr)

        # High-pass filtering
        y_filtered = highpass_filter(y_denoised, sr)

        # Save temporary file for pydub processing
        temp_wav_path = output_file.replace(".wav", "_temp.wav")
        sf.write(temp_wav_path, y_filtered, sr)

        # Load with pydub for normalization and compression
        audio = AudioSegment.from_wav(temp_wav_path)

        # Normalize volume
        normalized_audio = audio.apply_gain(-audio.max_dBFS)

        # Dynamic range compression (for mic clipping issues)
        compressed_audio = normalized_audio.compress_dynamic_range()

        # Save the final cleaned audio
        compressed_audio.export(output_file, format="wav")

        # Remove temporary file
        os.remove(temp_wav_path)

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def process_directory(input_dir, output_dir):
    """
    Process all WAV files in a directory.

    Args:
        input_dir (str): Directory containing original WAV files.
        output_dir (str): Directory to save cleaned WAV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    file_pairs = [(os.path.join(input_dir, file), os.path.join(output_dir, file)) for file in files]
    p_map(lambda x: process_audio(*x), file_pairs, num_cpus=os.cpu_count())

    print(f"Finished processing! Cleaned files saved in: {output_dir}")

# Example usage
input_directory = "/home/andrei/facultate/licenta/echo/enrolment_dataset"       # Folder with noisy audio files
output_directory = "/home/andrei/facultate/licenta/noisereduction_echo"  # Folder to save cleaned audio files

process_directory(input_directory, output_directory)
