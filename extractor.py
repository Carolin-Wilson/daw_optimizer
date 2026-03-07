import os
import reapy
import librosa
import numpy as np

def extract_audio():
    project = reapy.Project()
    # Create a folder for the snippets
    output_dir = os.path.join(os.getcwd(), "temp_audio")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Project: {project.name}")
    
    for track in project.tracks:
        # Skip empty tracks
        if not track.items:
            continue
            
        track_name = track.name if track.name else f"Track_{track.index + 1}"
        file_path = os.path.join(output_dir, f"track_{track.index}.wav")
        
        # This is a 'Dummy' render for now to test logic
        # In the next step, we will add the real render command
        print(f"Found track: {track_name} -> Preparing to analyze...")
        
def classify_spectral_centroid(path: str) -> str:
    """
    Load a WAV file, compute its average spectral centroid (Hz),
    and return 'BASS' if < 500 Hz or 'TREBLE' if > 3000 Hz.
    Otherwise returns 'UNCLASSIFIED'.
    """
    # Load audio; sr=None preserves the original sample rate
    y, sr = librosa.load(path, sr=None, mono=True)

    # Spectral centroid over time (shape: (1, n_frames))
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Average centroid in Hz
    mean_centroid_hz = float(centroids.mean())

    if mean_centroid_hz < 500.0:
        return "BASS"
    if mean_centroid_hz > 3000.0:
        return "TREBLE"

    # Between 500 Hz and 3000 Hz – not specified in your rule
    return "UNCLASSIFIED"

if __name__ == "__main__":
    extract_audio()