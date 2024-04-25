import time
import numpy as np
import torch.nn as nn
import torch
import pyaudio
import wave
from scipy.io import wavfile
from scipy.io import loadmat
from python_speech_features import mfcc
import scipy.signal as signal
import os
import random

# Include definitions of functions and classes here (set_seed, LSTMSoundClassifier, apply_lowpass_filter, preprocess_audio, etc.)
def set_seed(seed_value=42):
    """Fix the random number generator seed for all modules."""
    random.seed(seed_value)  # Built-in Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch function for CPU
    torch.cuda.manual_seed(seed_value)  # PyTorch function for GPU
    torch.cuda.manual_seed_all(seed_value)  # PyTorch function for multi-GPU
    torch.backends.cudnn.deterministic = True  # Makes cudnn's operation deterministic
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)

class LSTMSoundClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=3):
        super(LSTMSoundClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    

def apply_lowpass_filter(input_audio, b, a):
    filtered_audio = signal.lfilter(b, a, input_audio)
    return filtered_audio

def preprocess_audio(file_path, b, a):
    Fs, input_audio = wavfile.read(file_path)
    filtered_audio = apply_lowpass_filter(input_audio, b, a)
    mfcc_features = mfcc(filtered_audio, samplerate=Fs, numcep=13)
    return mfcc_features

def analyze_audio_file(file_path, model, b, a, index_to_labels, threshold):
    test_feature = preprocess_audio(file_path, b, a)
    test_feature_tensor = torch.tensor([test_feature], dtype=torch.float)
    with torch.no_grad():
        output = model(test_feature_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()
        predicted_prob = probabilities.max(dim=1).values.item()
        if predicted_prob < threshold:
            return None, None  # Indicates a non-match
        else:
            predicted_label = index_to_labels[predicted_index]
            return predicted_label, predicted_prob

def record_audio(duration=3, filename="recorded_audio.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = duration

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    set_seed(42)  # Fix seed

    labels_index = {
    'Velvet_Scoter_Melanitta_fusca': 0,
    'Long_tailed_Duck_Clangula_hyemalis': 1,
    'Leach\'s_Storm_Petrel_Hydrobates_leucorhous': 2,
    'Horned_Grebe_Podiceps_auritus': 3,
    'Great_Bustard_Otis_tarda': 4,
    'European_Turtle_Dove_Streptopelia_turtur': 5,
    'Black_legged_Kittiwake_Rissa_tridactyla': 6,
    'Balearic_Shearwater_Puffinus_mauretanicus': 7,
    'Atlantic_Puffin_Fratercula_arctica': 8,
    'Aquatic_Warbler_Acrocephalus_paludicola': 9
    }
    index_to_labels = {v: k for k, v in labels_index.items()}
    threshold = 0.9  # Set a threshold

    model = LSTMSoundClassifier(input_size=13, hidden_size=512, num_classes=len(labels_index), num_layers=3)
    model.load_state_dict(torch.load('3.23.pth', map_location=torch.device('cpu')))
    model.eval()

    filter_data = loadmat('highpass(500).mat')
    Coeffs = filter_data['ba'].astype(np.float64)
    b = Coeffs[0, :]
    a = 1

    labels_count = {label: 0 for label in labels_index.keys()}
    start_time = time.time()

    try:
        while True:
            record_audio(duration=3, filename="recorded_audio.wav")
            predicted_label, predicted_prob = analyze_audio_file("recorded_audio.wav", model, b, a, index_to_labels, threshold)
            if predicted_label:
                labels_count[predicted_label] += 1
            
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            print(f"Current counts: {labels_count}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            # Add a delay if needed, e.g., time.sleep(1)

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")

    # Final output
    end_time = time.time()
    total_time = end_time - start_time
    print("Final counts:")
    for label, count in labels_count.items():
        print(f"{label}: {count}")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()

