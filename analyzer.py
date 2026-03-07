import os
import reapy
import librosa
import numpy as np
import soundfile as sf


def analyze_and_organize(rename=True):
    project = reapy.Project()
    print(f"🔍 Deep Scanning {len(project.tracks)} tracks in REAPER...\n")

    for track in project.tracks:
        if len(track.items) == 0:
            continue

        try:
            # 1. Grab the audio file
            item = track.items[0]
            take = item.active_take
            filepath = take.source.filename

            # 2. Listen and Extract Features
            y, sr = librosa.load(filepath, duration=2.0)

            # Feature 1: Pitch
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

            # Feature 2: Noise level
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))

            print(f"🎵 Analyzing track {track.index + 1}...")
            print(f"   Pitch: {centroid:.0f}Hz | Noise (ZCR): {zcr:.3f}")

            # 3. Instrument Decision Logic
            if centroid < 1000 and zcr < 0.05:
                instrument = "KICK_BASS"
                color = (255, 0, 0)

            elif centroid > 4000 or zcr > 0.08:
                instrument = "HIHAT_CYMBAL"
                color = (0, 200, 255)

            elif 1000 <= centroid <= 4000 and zcr > 0.04:
                instrument = "SNARE_CLAP"
                color = (0, 255, 0)

            else:
                instrument = "SYNTH_VOCAL"
                color = (150, 0, 255)

            # 4. Rename Track (if checkbox enabled)
            if rename:
                track.name = f"{instrument}_{track.index + 1}"

            # 5. Color Track
            track.color = color

            print(f"   ✅ Identified as: {instrument} | Track Updated!\n")

        except Exception as e:
            print(f"⚠️ Error on {track.name}: {e}")

    print("🎉 Full DAW Organization Complete! Check REAPER.")


# -------------------------------------------------------
# Function Flask will call
# -------------------------------------------------------

def run_optimization(target, num_stems, basis, rename):
    print("\n⚙️ Running Optimization Pipeline...")
    print(f"Target: {target}")
    print(f"Number of Stems: {num_stems}")
    print(f"Basis: {basis}")
    print(f"Rename Tracks: {rename}\n")

    # Right now we just run the analyzer
    analyze_and_organize(rename) 

def export_tracks(target, num_stems, basis):
    project = reapy.Project()

    print("📦 Starting export process...")

    if target == "all":
        tracks = project.tracks
    elif target == "selected":
        tracks = project.selected_tracks
    else:
        tracks = project.tracks

    export_folder = "exports"
    os.makedirs(export_folder, exist_ok=True)

    for track in tracks:
        if len(track.items) == 0:
            print(f"⏭️ Skipping {track.name} (no items)")
            continue

        item = track.items[0]
        take = item.active_take
        source_path = take.source.filename

        if not source_path or not os.path.isfile(source_path):
            print(f"⚠️ Skipping {track.name}: source file not found")
            continue

        out_path = os.path.join(export_folder, f"{track.name}.wav")
        print(f"Exporting {track.name} → {out_path}")

        y, sr = librosa.load(source_path, sr=None, mono=False)
        sf.write(out_path, y.T if y.ndim > 1 else y, sr)

    print("✅ Export finished!")