import csv
import os
import re
import reapy
import librosa
import numpy as np
import soundfile as sf

# YAMNet sample rate (fixed)
YAMNET_SR = 16000

# Lazy-loaded YAMNet model and class names
_yamnet_model = None
_yamnet_class_names = None

# Stem-type colors: map YAMNet class display_name (lower) to (R, G, B)
# Palette based on frequency bands:
# - Sub Bass (20–80 Hz): deep red / burgundy
# - Bass (80–250 Hz): orange / brown
# - Mid (250–2000 Hz): blue / indigo
# - Presence (2000–8000 Hz): green / teal
# - Air (8000 Hz+): light grey / white
SUB_BASS_COLOR = (128, 0, 32)     # deep red / burgundy
BASS_COLOR = (205, 133, 63)       # orange / brown
MID_COLOR = (65, 105, 225)        # blue / indigo
PRESENCE_COLOR = (0, 128, 128)    # green / teal
AIR_COLOR = (230, 230, 230)       # light grey

_STEM_COLORS = {
    # Sub bass / bass elements
    "kick": SUB_BASS_COLOR,
    "bass drum": SUB_BASS_COLOR,
    "bass guitar": BASS_COLOR,
    "drum": BASS_COLOR,
    # Mid instruments
    "snare drum": MID_COLOR,
    "clapping": MID_COLOR,
    "clap": MID_COLOR,
    "piano": MID_COLOR,
    "guitar": MID_COLOR,
    "vocal": MID_COLOR,
    "singing": MID_COLOR,
    # Presence / air
    "synthesizer": PRESENCE_COLOR,
    "cymbal": PRESENCE_COLOR,
    "hi-hat": AIR_COLOR,
}
_DEFAULT_COLOR = AIR_COLOR


_INSTRUMENT_KEYWORDS = [
    "drum",
    "snare",
    "bass drum",
    "hi-hat",
    "cymbal",
    "tom",
    "percussion",
    "guitar",
    "bass guitar",
    "piano",
    "keyboard",
    "synthesizer",
    "organ",
    "violin",
    "cello",
    "string",
    "saxophone",
    "trumpet",
    "horn",
    "flute",
    "vocal",
    "singing",
    "voice",
    "choir",
]

_GENERIC_MUSIC_LABELS = {
    "Music",
    "Pop music",
    "Rock music",
    "Electronic music",
    "Classical music",
    "Soundtrack music",
    "Background music",
    "Video game music",
    "Dance music",
    "Jazz",
    "Hip hop music",
    "Reggae",
    "Blues",
    "Country",
    "Funk",
    "Disco",
    "Ska",
    "House music",
    "Techno",
    "Trance music",
    "Ambient music",
}


def _sanitize_track_name(name):
    """Make a string safe for use as a filename (no path chars, no spaces)."""
    if not name or not str(name).strip():
        return "Track"
    s = re.sub(r'[^\w\-.]', '_', str(name).strip())
    return s.strip("_") or "Track"


def _load_yamnet():
    global _yamnet_model, _yamnet_class_names
    if _yamnet_model is not None:
        return
    import tensorflow_hub as hub
    _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    _yamnet_class_names = _get_yamnet_class_names(_yamnet_model)


def _get_yamnet_class_names(model):
    """Get list of 521 class display names from model or local CSV."""
    try:
        class_map_path = model.class_map_path().numpy().decode("utf-8")
        if os.path.isfile(class_map_path):
            return _class_names_from_csv_path(class_map_path)
    except Exception:
        pass
    local_csv = os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv")
    if os.path.isfile(local_csv):
        return _class_names_from_csv_path(local_csv)
    return [f"Class_{i}" for i in range(521)]


def _class_names_from_csv_path(path):
    names = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            # display_name is 3rd column; may contain commas
            name = ",".join(row[2:]).strip() if len(row) > 2 else ""
            names.append(name)
    return names if len(names) == 521 else [f"Class_{i}" for i in range(521)]


def _color_for_class(display_name):
    """Return (R, G, B) for a YAMNet display_name."""
    key = (display_name or "").lower()
    for stem_key, color in _STEM_COLORS.items():
        if stem_key in key:
            return color
    return _DEFAULT_COLOR


def _is_instrument_label(name):
    """Return True if this YAMNet label looks like a specific instrument, not just 'Music'."""
    if not name:
        return False
    if name in _GENERIC_MUSIC_LABELS:
        return False
    lower = name.lower()
    return any(keyword in lower for keyword in _INSTRUMENT_KEYWORDS)


def _bucket_from_yamnet_scores(scores, y, sr):
    """
    Map YAMNet class scores + basic spectral features to our stem buckets.

    Returns one of:
        KICK_BASS, SNARE_CLAP, HIHAT_CYMBAL,
        BASS, GUITAR, KEYS, VOCAL, SYNTH,
        DRUM_MISC, or None if uncertain.
    """
    scores = np.array(scores)
    if scores.ndim > 1:
        scores = scores.mean(axis=0)

    indices = np.argsort(scores)[::-1]
    top_indices = indices[:30]
    top_names = [
        _yamnet_class_names[i].lower() if i < len(_yamnet_class_names) else ""
        for i in top_indices
    ]

    def has(word):
        return any(word in name for name in top_names)

    # Basic spectral features to split drum roles
    centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    )
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Drum / percussion family first
    if has("drum") or has("percussion") or has("snare drum") or has("bass drum"):
        if centroid < 1000 and zcr < 0.05:
            return "KICK_BASS"
        if centroid > 4000 or zcr > 0.08:
            return "HIHAT_CYMBAL"
        if 1000 <= centroid <= 4000 and zcr > 0.04:
            return "SNARE_CLAP"
        return "DRUM_MISC"

    # Bass instruments
    if has("bass guitar") or has("double bass") or has("bass"):
        return "BASS"

    # Guitars / plucked strings
    if has("guitar"):
        return "GUITAR"

    # Keys / keyboards
    if has("piano") or has("keyboard") or has("organ"):
        return "KEYS"

    # Vocals
    if has("vocal") or has("voice") or has("singing") or has("choir"):
        return "VOCAL"

    # Synth / electronic
    if has("synthesizer") or has("synth") or has("sampler") or has("electronic"):
        return "SYNTH"

    return None


def _classify_audio_yamnet(filepath):
    """
    Use YAMNet scores + simple heuristics to get an instrument-style label.

    Returns (label, top_class_index_from_yamnet).
    """
    _load_yamnet()

    y, sr = librosa.load(filepath, sr=YAMNET_SR, mono=True, duration=3.0)
    y = y.astype(np.float32)

    scores, _, _ = _yamnet_model(y)  # (frames, num_classes)

    # Try to map into our own stem bucket names first.
    bucket = _bucket_from_yamnet_scores(scores, y, sr)

    scores_arr = np.array(scores)
    if scores_arr.ndim > 1:
        scores_arr = scores_arr.mean(axis=0)

    top_idx = int(np.argmax(scores_arr))
    top_name = (
        _yamnet_class_names[top_idx]
        if top_idx < len(_yamnet_class_names)
        else f"Class_{top_idx}"
    )

    label = bucket or top_name
    return label, top_idx


def analyze_and_organize(rename=True):
    project = reapy.Project()
    print(f"🔍 Deep Scanning {len(project.tracks)} tracks in REAPER...\n")

    for track in project.tracks:
        if len(track.items) == 0:
            continue

        try:
            item = track.items[0]
            take = item.active_take
            filepath = take.source.filename

            display_name, class_idx = _classify_audio_yamnet(filepath)
            instrument_safe = _sanitize_track_name(display_name)
            color = _color_for_class(display_name)

            print(f"🎵 Analyzing track {track.index + 1}...")
            print(f"   YAMNet: {display_name} (class {class_idx})")

            if rename:
                track.name = f"{instrument_safe}_{track.index + 1}"
            track.color = color

            print(f"   ✅ Identified as: {display_name} | Track Updated!\n")

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

        safe_name = _sanitize_track_name(track.name)
        out_path = os.path.join(export_folder, f"{safe_name}.wav")
        print(f"Exporting {track.name} → {out_path}")

        y, sr = librosa.load(source_path, sr=None, mono=False)
        sf.write(out_path, y.T if y.ndim > 1 else y, sr)

    print("✅ Export finished!")