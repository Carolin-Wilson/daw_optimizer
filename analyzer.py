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
_yamnet_model       = None
_yamnet_class_names = None

# Lazy-loaded PANNs CNN14 model (optional – needs: pip install panns-inference)
_panns_model  = None
_panns_labels = None

# CREPE flag (optional – needs: pip install crepe)
_CREPE_AVAILABLE = None   # None = unchecked, True/False after first attempt

# ── Classification cache: filepath → label
# Populated by analyze_and_organize() and used by export_tracks() so that
# the export always knows the correct stem category even when rename=False.
_track_labels: dict = {}   # {source_filepath: label_string}

# ── Ensemble weights (must sum to 1.0)
# Adjust these if one model consistently outperforms the others.
WEIGHT_YAMNET   = 0.40   # YAMNet Google/AudioSet
WEIGHT_PANNS    = 0.35   # PANNs CNN14 / AudioSet (0 when not installed)
WEIGHT_SPECTRAL = 0.25   # Spectral + F0 heuristics

# -------------------------------------------------------
# Color palette
# -------------------------------------------------------
SUB_BASS_COLOR = (128, 0, 32)     # deep red / burgundy
BASS_COLOR = (205, 133, 63)       # orange / brown
MID_COLOR = (65, 105, 225)        # blue / indigo
PRESENCE_COLOR = (0, 128, 128)    # green / teal
AIR_COLOR = (230, 230, 230)       # light grey

_STEM_COLORS = {
    "kick":       SUB_BASS_COLOR,
    "bass drum":  SUB_BASS_COLOR,
    "bass guitar": BASS_COLOR,
    "bass":       BASS_COLOR,
    "drum":       BASS_COLOR,
    "snare drum": MID_COLOR,
    "clapping":   MID_COLOR,
    "clap":       MID_COLOR,
    "piano":      MID_COLOR,
    "guitar":     MID_COLOR,
    "vocal":      MID_COLOR,
    "singing":    MID_COLOR,
    "synthesizer": PRESENCE_COLOR,
    "cymbal":     PRESENCE_COLOR,
    "hi-hat":     AIR_COLOR,
}
_DEFAULT_COLOR = AIR_COLOR

# Label → color lookup (used for our bucket labels too)
_BUCKET_COLORS = {
    "KICK_BASS":    SUB_BASS_COLOR,
    "BASS":         BASS_COLOR,
    "SNARE_CLAP":   MID_COLOR,
    "DRUM_MISC":    BASS_COLOR,
    "GUITAR":       MID_COLOR,
    "KEYS":         MID_COLOR,
    "VOCAL":        MID_COLOR,
    "SYNTH":        PRESENCE_COLOR,
    "HIHAT_CYMBAL": AIR_COLOR,
    "WIND":         PRESENCE_COLOR,
    "SILENCE":      (40, 40, 40),      # very dark grey
    "UNKNOWN":      (100, 100, 100),   # grey
}

_INSTRUMENT_KEYWORDS = [
    "drum", "snare", "bass drum", "hi-hat", "cymbal", "tom", "percussion",
    "guitar", "bass guitar", "piano", "keyboard", "synthesizer", "organ",
    "violin", "cello", "string", "saxophone", "trumpet", "horn", "flute",
    "vocal", "singing", "voice", "choir",
]

_GENERIC_MUSIC_LABELS = {
    "Music", "Pop music", "Rock music", "Electronic music", "Classical music",
    "Soundtrack music", "Background music", "Video game music", "Dance music",
    "Jazz", "Hip hop music", "Reggae", "Blues", "Country", "Funk", "Disco",
    "Ska", "House music", "Techno", "Trance music", "Ambient music",
}

# -------------------------------------------------------
# Silence / RMS threshold
# -------------------------------------------------------
# Frames whose RMS is below this fraction of the track's peak RMS are silence.
_SILENCE_RMS_RATIO = 0.02   # 2 % of peak → treat as silent
_MIN_ACTIVE_SECONDS = 0.5   # we need at least this much non-silent audio


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


def _load_panns():
    """
    Lazy-load PANNs CNN14 AudioTagging model.
    Silently skips if panns-inference is not installed.
    After first call, _panns_model is either loaded or the string "UNAVAILABLE".
    """
    global _panns_model, _panns_labels
    if _panns_model is not None:
        return
    try:
        from panns_inference import AudioTagging, labels as panns_label_list
        print("[Ensemble] Loading PANNs CNN14 model...")
        _panns_model  = AudioTagging(checkpoint_path=None, device='cpu')
        _panns_labels = list(panns_label_list)
        print(f"[Ensemble] PANNs loaded ({len(_panns_labels)} classes).")
    except ImportError:
        print("[Ensemble] panns-inference not installed — skipping PANNs. "
              "Install with: pip install panns-inference")
        _panns_model  = "UNAVAILABLE"
        _panns_labels = []
    except Exception as exc:
        print(f"[Ensemble] PANNs load failed: {exc}")
        _panns_model  = "UNAVAILABLE"
        _panns_labels = []


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
            name = ",".join(row[2:]).strip() if len(row) > 2 else ""
            names.append(name)
    return names if len(names) == 521 else [f"Class_{i}" for i in range(521)]


def _color_for_label(label):
    """Return (R, G, B) for a bucket label or YAMNet display_name."""
    if label in _BUCKET_COLORS:
        return _BUCKET_COLORS[label]
    key = (label or "").lower()
    for stem_key, color in _STEM_COLORS.items():
        if stem_key in key:
            return color
    return _DEFAULT_COLOR


def _is_instrument_label(name):
    """Return True if this YAMNet label looks like a specific instrument."""
    if not name:
        return False
    if name in _GENERIC_MUSIC_LABELS:
        return False
    lower = name.lower()
    return any(keyword in lower for keyword in _INSTRUMENT_KEYWORDS)


# -------------------------------------------------------
# Silence-aware audio loading
# -------------------------------------------------------

def _load_active_segment(filepath, sr=YAMNET_SR, target_duration=4.0,
                         max_scan_secs=180.0):
    """
    Load ``target_duration`` seconds of *active* (non-silent) audio.

    Strategy
    --------
    1. Load up to ``max_scan_secs`` of the file at 8 kHz (cheap probe) to find
       the first non-silent region quickly.
    2. librosa.effects.split() returns SAMPLE indices — divide by probe_sr to
       get seconds.
    3. Return a ``target_duration``-second clip from that offset.
       Falls back to the raw file start if everything is silent.
    """
    probe_sr  = 8000           # cheap probe sample rate
    probe_dur = min(max_scan_secs, 300.0)

    try:
        y_probe, _ = librosa.load(
            filepath, sr=probe_sr, mono=True, duration=probe_dur
        )
    except Exception:
        y, _ = librosa.load(filepath, sr=sr, mono=True, duration=target_duration)
        return y.astype(np.float32)

    if y_probe.size == 0:
        y, _ = librosa.load(filepath, sr=sr, mono=True, duration=target_duration)
        return y.astype(np.float32)

    # Find non-silent intervals — try two thresholds
    intervals = librosa.effects.split(y_probe, top_db=40)
    if len(intervals) == 0:
        intervals = librosa.effects.split(y_probe, top_db=55)

    if len(intervals) == 0:
        print("   ⚠️  No active audio found in probe — using file start.")
        offset = 0.0
    else:
        # librosa.effects.split() returns SAMPLE indices (not frame indices).
        # offset_seconds = sample_index / sample_rate  ← correct conversion
        first_active_sample = int(intervals[0][0])
        offset = float(first_active_sample) / probe_sr
        print(f"   ⏩ Silence skip: {offset:.2f}s")

    # Now load the real audio starting at the detected offset
    y, _ = librosa.load(
        filepath, sr=sr, mono=True, offset=offset, duration=target_duration
    )

    # Trim any residual leading silence after the seek
    # librosa.effects.trim returns (y_trimmed, trim_indices_array)
    if y.size > 0:
        y_trimmed, trim_idx = librosa.effects.trim(y, top_db=40)
        # Only use the trimmed version if it kept reasonable content
        if y_trimmed.size >= int(sr * _MIN_ACTIVE_SECONDS):
            y = y_trimmed

    # If still empty or very short, just grab from the very beginning
    if y.size < int(sr * _MIN_ACTIVE_SECONDS):
        print("   ℹ️  Active segment too short — loading from file start.")
        y, _ = librosa.load(filepath, sr=sr, mono=True, duration=target_duration)

    # ── Final silence gate: if RMS is essentially zero, the file is silent
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size > 0 else 0.0
    if rms < 1e-5:   # -100 dBFS — inaudible even at max gain
        print("   🔇 Audio is truly silent (RMS < 1e-5) — marking as SILENCE.")
        return None   # sentinel: caller must handle None

    return y.astype(np.float32)


# -------------------------------------------------------
# Spectral + temporal feature extraction
# -------------------------------------------------------

# -------------------------------------------------------
# Filename-based hint (highest priority)
# -------------------------------------------------------

_FILENAME_PATTERNS = [
    # Most specific first
    (r'\b(kick|bd|bassdrum|bass[-_.]?drum)\b',           'KICK_BASS'),
    (r'\b(snare|sd|clap|rimshot)\b',                     'SNARE_CLAP'),
    (r'\b(hihat|hi[-_.]?hat|hh|cymbal|crash|ride|overhead)\b', 'HIHAT_CYMBAL'),
    (r'\b(drum|perc|percussion|tom|loop)\b',             'DRUM_MISC'),
    (r'\b(bass|sub(?!string))\b',                        'BASS'),
    (r'\b(voc|vocal|vox|voice|sing|choir|lead|adlib|backing)\b', 'VOCAL'),
    (r'\b(gtr|guitar|git|strum|pick)\b',                 'GUITAR'),
    (r'\b(piano|keys|keyboard|organ|clav|rhodes|wurli)\b','KEYS'),
    (r'\b(synth|pad|arp|lead|osc|oscillator|electronic)\b','SYNTH'),
    (r'\b(flute|sax|saxophone|trumpet|trombone|horn|brass|violin|cello|string|wind|woodwind|clarinet|oboe|harp)\b', 'WIND'),
]


def _label_from_filename(filepath):
    """
    Try to infer the stem bucket from the filename / parent folder name.
    Checks the file stem and its immediate parent directory.
    Returns a bucket string or None.
    """
    if not filepath:
        return None
    parts = [
        os.path.splitext(os.path.basename(filepath))[0],  # filename without ext
        os.path.basename(os.path.dirname(filepath)),       # parent folder name
    ]
    combined = ' '.join(parts).lower()
    for pattern, label in _FILENAME_PATTERNS:
        if re.search(pattern, combined):
            return label
    return None


def _extract_features(y, sr):
    """
    Return a dict of audio features used for classification.

    Features
    --------
    centroid       – spectral centroid (Hz)
    zcr            – zero-crossing rate
    flatness       – spectral flatness (0=tonal, 1=noise-like)
    rms            – root-mean-square energy
    onset_rate     – onsets per second (useful: drums are dense)
    attack_time    – mean time to first peak per onset (short = percussive)
    sub_ratio      – energy ratio in 20–80 Hz band
    bass_ratio     – energy ratio in 80–250 Hz band
    mid_ratio      – energy ratio in 250–2 kHz band
    presence_ratio – energy ratio in 2–8 kHz band
    air_ratio      – energy ratio in 8+ kHz band
    mfcc           – first 13 MFCC means (np array)
    mfcc_delta_var – variance of ΔMFCC (measures timbral change over time)
    """
    n_fft = 2048
    hop   = 512
    feats = {}

    S     = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    S_pow = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # ── Spectral centroid / ZCR / flatness
    feats["centroid"] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    feats["zcr"]      = float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=hop)))
    feats["flatness"] = float(np.mean(librosa.feature.spectral_flatness(S=S_pow)))

    # ── RMS energy
    feats["rms"] = float(np.mean(librosa.feature.rms(y=y)))

    # ── Onset density
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
    duration_sec = len(y) / sr
    feats["onset_rate"] = len(onset_frames) / max(duration_sec, 0.1)

    # ── Attack time estimate (percussive = short attack)
    if len(onset_frames) > 0:
        attacks = []
        for onset_f in onset_frames[:8]:
            window = S[:, onset_f: onset_f + 10]
            if window.shape[1] > 1:
                energy = np.sum(window, axis=0)
                peak   = np.argmax(energy)
                attacks.append(peak * hop / sr)
        feats["attack_time"] = float(np.mean(attacks)) if attacks else 0.05
    else:
        feats["attack_time"] = 0.05

    # ── Band energy ratios (use power spectrum)
    total_energy = np.sum(S_pow) + 1e-9
    def _band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(S_pow[mask])) / total_energy

    feats["sub_ratio"]      = _band_energy(20,    80)
    feats["bass_ratio"]     = _band_energy(80,   250)
    feats["mid_ratio"]      = _band_energy(250,  2000)
    feats["presence_ratio"] = _band_energy(2000, 8000)
    feats["air_ratio"]      = _band_energy(8000, sr / 2)

    # ── Pitch / F0 estimation
    # Try CREPE first (neural, more accurate for melodic content).
    # Fall back to librosa.yin if CREPE is unavailable.
    global _CREPE_AVAILABLE
    crepe_f0, crepe_conf = None, 0.0

    if _CREPE_AVAILABLE is None:
        try:
            import crepe  # noqa: F401
            _CREPE_AVAILABLE = True
        except ImportError:
            _CREPE_AVAILABLE = False
            print("[Ensemble] crepe not installed — using librosa.yin for F0. "
                  "Install with: pip install crepe")

    if _CREPE_AVAILABLE:
        try:
            import crepe
            # Use 'tiny' capacity for speed; 'small'/'medium'/'full' for accuracy
            crp_time, crp_freq, crp_conf, _ = crepe.predict(
                y, sr, viterbi=True,
                step_size=20,
                model_capacity='tiny',
                verbose=0,
            )
            mask = (crp_conf > 0.5) & (crp_freq < 2100.0)
            if mask.sum() > 3:
                crepe_f0   = float(np.median(crp_freq[mask]))
                crepe_conf = float(mask.sum()) / max(len(mask), 1)
        except Exception as exc:
            print(f"[Ensemble] CREPE predict error: {exc}")

    if crepe_f0 is not None:
        feats["f0_median"]     = crepe_f0
        feats["f0_confidence"] = crepe_conf
    else:
        # Fallback: librosa YIN
        try:
            f0_yin = librosa.yin(
                y, fmin=40.0, fmax=2100.0,
                sr=sr, hop_length=hop, frame_length=n_fft,
            )
            f0_clean = f0_yin[f0_yin < 2000.0]
            if f0_clean.size > 5:
                feats["f0_median"]     = float(np.median(f0_clean))
                feats["f0_confidence"] = float(f0_clean.size) / float(f0_yin.size)
            else:
                feats["f0_median"]     = None
                feats["f0_confidence"] = 0.0
        except Exception:
            feats["f0_median"]     = None
            feats["f0_confidence"] = 0.0

    # ── Harmonic ratio: energy in harmonic partials vs total
    # Use librosa harmonic/percussive separation
    try:
        y_harm = librosa.effects.harmonic(y, margin=3.0)
        harm_rms  = float(np.mean(librosa.feature.rms(y=y_harm)))
        total_rms = float(np.mean(librosa.feature.rms(y=y))) + 1e-9
        feats["harmonic_ratio"] = min(harm_rms / total_rms, 1.0)
    except Exception:
        feats["harmonic_ratio"] = 0.5

    # ── MFCCs
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    except TypeError:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    feats["mfcc"]           = mfcc.mean(axis=1)
    feats["mfcc_delta_var"] = float(np.var(librosa.feature.delta(mfcc)))

    return feats


# -------------------------------------------------------
# YAMNet + heuristic classifier
# -------------------------------------------------------

def _bucket_from_scores_and_features(scores, feats):
    """
    Combine YAMNet scores + rich spectral/pitch features → stem bucket.

    Decision order:
        0. Silence gate (centroid=0, all bands=0, YAMNet says Silence)
        1. Percussion family  (drums, hats, snare)
        2. Bass               (low F0 + low-band energy)
        3. Vocal              (mid F0 + formant energy)
        4. Wind/Strings       (very tonal, specific F0 range)
        5. Guitar             (plucked-string timbre)
        6. Keys               (piano / organ)
        7. Synth              (electronic)
        8. Fallback: dominant band
    """
    scores = np.array(scores)
    if scores.ndim > 1:
        scores = scores.mean(axis=0)

    # Score ALL 521 classes for maximum YAMNet signal
    all_names  = [n.lower() if n else "" for n in (_yamnet_class_names or [])]
    all_scores = scores.tolist()

    def ev(*words):
        """Sum YAMNet scores across ALL classes whose name contains any keyword."""
        total = 0.0
        for name, sc in zip(all_names, all_scores):
            if any(w in name for w in words):
                total += sc
        return total

    # Feature aliases
    centroid       = feats["centroid"]
    zcr            = feats["zcr"]
    flatness       = feats["flatness"]
    onset_rate     = feats["onset_rate"]
    attack_time    = feats["attack_time"]
    sub_ratio      = feats["sub_ratio"]
    bass_ratio     = feats["bass_ratio"]
    mid_ratio      = feats["mid_ratio"]
    presence_ratio = feats["presence_ratio"]
    air_ratio      = feats["air_ratio"]
    f0             = feats["f0_median"]       # None = unvoiced / noise
    f0_conf        = feats["f0_confidence"]
    harm_ratio     = feats["harmonic_ratio"]
    rms            = feats["rms"]

    # ─────────────────────────────────────────────
    # 0. SILENCE GATE
    # If the audio is below noise floor, YAMNet confirms silence,
    # or all spectral features are zero, return SILENCE immediately.
    # ─────────────────────────────────────────────
    yamnet_silence_score = ev("silence")      # YAMNet's own silence class
    total_band_energy    = sub_ratio + bass_ratio + mid_ratio + presence_ratio + air_ratio
    is_silent = (
        rms < 5e-4                             # very low energy
        or centroid < 1.0                      # spectral centroid essentially 0
        or total_band_energy < 0.01            # no meaningful band content
        or yamnet_silence_score > 0.50         # YAMNet strongly says silence
    )
    if is_silent:
        return "SILENCE"

    # ── YAMNet category evidence
    ev_drum   = ev("drum", "percussion", "kick", "snare", "tom",
                   "bass drum", "snare drum", "hi-hat")
    ev_bass   = ev("bass guitar", "double bass", "upright bass", "bass")
    ev_guitar = ev("guitar", "acoustic guitar", "electric guitar",
                   "plucked string", "banjo", "ukulele")
    ev_keys   = ev("piano", "keyboard", "organ", "harpsichord",
                   "electric piano", "synthesizer keyboard")
    ev_vocal  = ev("vocal", "voice", "singing", "choir",
                   "speech", "human voice", "a cappella", "humming")
    ev_synth  = ev("synthesizer", "synth", "electronic",
                   "sampler", "theremin")
    ev_hihat  = ev("hi-hat", "cymbal", "crash cymbal",
                   "ride cymbal", "shaker")
    ev_wind   = ev("flute", "saxophone", "trumpet", "trombone",
                   "clarinet", "oboe", "horn", "brass",
                   "violin", "cello", "string", "fiddle",
                   "woodwind", "reed", "wind instrument", "bowed")

    # ─────────────────────────────────────────────
    # 1. PERCUSSION
    # Percussive signals: high onset rate, very short attack, or
    # clear drum YAMNet evidence.  Harmonic ratio is LOW for drums.
    # ─────────────────────────────────────────────
    is_percussive = (
        ev_drum > 0.06
        or (onset_rate > 7.0 and harm_ratio < 0.55)
        or (attack_time < 0.012 and zcr > 0.06 and harm_ratio < 0.50)
    )

    if is_percussive:
        kick_score = (
            (sub_ratio + bass_ratio)
            - presence_ratio - air_ratio
            + (0.15 if centroid < 300 else 0)
            + (0.10 if attack_time < 0.010 else 0)
            + (0.10 if zcr < 0.07 else 0)
            - (0.20 if harm_ratio > 0.6 else 0)   # tonal → not kick
        )
        snare_score = (
            mid_ratio
            + (0.15 if 1200 < centroid < 6000 else 0)
            + (0.15 if zcr > 0.08 else 0)
            + (0.10 if 0.2 < flatness < 0.7 else 0)
        )
        hihat_score = (
            presence_ratio + air_ratio
            + (0.20 if centroid > 5000 else 0)
            + (0.20 if zcr > 0.12 else 0)
            + (0.10 if onset_rate > 8 else 0)
            + ev_hihat * 2
        )

        best = max(kick_score, snare_score, hihat_score)
        if best == kick_score and kick_score > 0.12:
            return "KICK_BASS"
        if best == hihat_score and hihat_score > 0.18:
            return "HIHAT_CYMBAL"
        if best == snare_score and snare_score > 0.12:
            return "SNARE_CLAP"
        if centroid > 2000 and attack_time < 0.008 and zcr > 0.10:
            return "SNARE_CLAP"
        return "DRUM_MISC"

    # ─────────────────────────────────────────────
    # 2. BASS
    # Key: F0 below ~280 Hz, or strong sub/bass band energy with tonal quality.
    # Bass guitar harmonics can bleed into mid-range, so F0 is decisive.
    # ─────────────────────────────────────────────
    if ev_bass > 0.06:
        return "BASS"
    if (
        (sub_ratio + bass_ratio) > 0.42
        and centroid < 700
        and harm_ratio > 0.35          # tonal (not noise / kick)
        and onset_rate < 12
        and (f0 is None or f0 < 280)   # F0 in bass range when measurable
    ):
        return "BASS"
    # F0 strongly says bass even if band energy is spread
    if f0 is not None and f0 < 200 and f0_conf > 0.4 and harm_ratio > 0.4:
        return "BASS"

    # ─────────────────────────────────────────────
    # 3. VOCAL
    # Key: F0 in human vocal range (100–900 Hz), mid-range energy,
    # moderate harmonic ratio, and NOT bass-dominant.
    # ─────────────────────────────────────────────
    if ev_vocal > 0.07:
        return "VOCAL"
    vocal_f0_ok = (f0 is not None and 100 < f0 < 900 and f0_conf > 0.35)
    if vocal_f0_ok and mid_ratio > 0.30 and (sub_ratio + bass_ratio) < 0.45:
        return "VOCAL"
    if (
        ev_vocal > 0.03
        and mid_ratio > 0.35
        and 600 < centroid < 4000
        and harm_ratio > 0.45
        and (sub_ratio + bass_ratio) < 0.40
    ):
        return "VOCAL"

    # ─────────────────────────────────────────────
    # 4. WIND / STRINGS
    # Flutes, sax, trumpet, violin etc.: very high harmonic purity,
    # clear F0, sustained long notes (low onset rate), specific F0 ranges.
    # ─────────────────────────────────────────────
    if ev_wind > 0.05:
        return "WIND"
    wind_f0_ok = (f0 is not None and 230 < f0 < 2100 and f0_conf > 0.4)
    if (
        wind_f0_ok
        and harm_ratio > 0.65          # very tonal — near-sine
        and flatness < 0.10            # low spectral flatness = tonal
        and onset_rate < 5.0           # sustained notes
        and (sub_ratio + bass_ratio) < 0.30
    ):
        return "WIND"

    # ─────────────────────────────────────────────
    # 5. GUITAR
    # Plucked string: moderate F0 range, more transients than wind,
    # broader harmonic spectrum than flute.
    # ─────────────────────────────────────────────
    if ev_guitar > 0.05:
        return "GUITAR"
    guitar_f0_ok = (f0 is not None and 70 < f0 < 1400)
    if (
        (ev_guitar > 0.02 or guitar_f0_ok)
        and (bass_ratio + mid_ratio) > 0.45
        and centroid < 4000
        and 0.05 < flatness < 0.50     # not too noisy, not pure sine
        and harm_ratio > 0.40
        and onset_rate < 14
    ):
        return "GUITAR"

    # ─────────────────────────────────────────────
    # 6. KEYS
    # ─────────────────────────────────────────────
    if ev_keys > 0.05:
        return "KEYS"
    if (
        ev_keys > 0.02
        and mid_ratio > 0.38
        and flatness < 0.25
        and 300 < centroid < 4000
        and harm_ratio > 0.50
    ):
        return "KEYS"

    # ─────────────────────────────────────────────
    # 7. SYNTH / ELECTRONIC
    # ─────────────────────────────────────────────
    if ev_synth > 0.05:
        return "SYNTH"
    if ev_synth > 0.02 and (mid_ratio + presence_ratio) > 0.55:
        return "SYNTH"

    # ─────────────────────────────────────────────
    # 8. Fallback: dominant frequency band
    # ─────────────────────────────────────────────
    dominant = max(
        ("SUB_BASS",  sub_ratio),
        ("BASS_BAND", bass_ratio),
        ("MID",       mid_ratio),
        ("PRESENCE",  presence_ratio),
        ("AIR",       air_ratio),
        key=lambda x: x[1],
    )
    band_fallback = {
        "SUB_BASS":  "KICK_BASS",
        "BASS_BAND": "BASS",
        "MID":       "GUITAR",
        "PRESENCE":  "SYNTH",
        "AIR":       "HIHAT_CYMBAL",
    }
    return band_fallback.get(dominant[0])


# -------------------------------------------------------
# Per-model bucket score functions  (each returns {bucket: float})
# -------------------------------------------------------

_ALL_BUCKETS = [
    "KICK_BASS", "SNARE_CLAP", "HIHAT_CYMBAL", "DRUM_MISC",
    "BASS", "VOCAL", "WIND", "GUITAR", "KEYS", "SYNTH", "SILENCE",
]

# Keyword groups that map to each bucket  (used by both YAMNet & PANNs)
_BUCKET_KEYWORDS = {
    "KICK_BASS":    ["kick", "bass drum", "bass_drum", "kick drum"],
    "SNARE_CLAP":   ["snare", "clap", "rimshot", "hand clap"],
    "HIHAT_CYMBAL": ["hi-hat", "hihat", "cymbal", "crash", "ride cymbal", "shaker"],
    "DRUM_MISC":    ["drum kit", "drum", "percussion", "tom", "bongo", "conga", "cowbell"],
    "BASS":         ["bass guitar", "double bass", "upright bass", "electric bass"],
    "VOCAL":        ["singing", "vocal", "voice", "choir", "a cappella",
                     "humming", "speech", "human voice"],
    "WIND":         ["flute", "saxophone", "trumpet", "trombone", "clarinet",
                     "oboe", "horn", "brass", "violin", "cello",
                     "bowed string", "wind instrument", "woodwind", "fiddle"],
    "GUITAR":       ["guitar", "acoustic guitar", "electric guitar",
                     "plucked string", "banjo", "ukulele", "mandolin"],
    "KEYS":         ["piano", "keyboard", "organ", "harpsichord",
                     "electric piano", "accordion"],
    "SYNTH":        ["synthesizer", "electronic music", "sampler",
                     "theremin", "electronic"],
    "SILENCE":      ["silence", "silent"],
}


def _scores_to_bucket_votes(class_names, class_scores):
    """
    Given a list of class display names and matching float scores,
    return a {bucket: float} dict where each bucket accumulates the
    sum of scores from every class whose name matches a bucket keyword.
    """
    all_lower = [n.lower() if n else "" for n in class_names]
    votes = {b: 0.0 for b in _ALL_BUCKETS}
    for name, sc in zip(all_lower, class_scores):
        for bucket, keywords in _BUCKET_KEYWORDS.items():
            if any(kw in name for kw in keywords):
                votes[bucket] += float(sc)
                break   # each class counts toward at most one bucket
    return votes


def _yamnet_bucket_votes(yamnet_scores):
    """Convert raw YAMNet scores to {bucket: float}."""
    scores = np.array(yamnet_scores)
    if scores.ndim > 1:
        scores = scores.mean(axis=0)
    return _scores_to_bucket_votes(
        _yamnet_class_names or [], scores.tolist()
    )


def _panns_bucket_votes(y):
    """
    Run PANNs CNN14 on audio y (float32, 32 kHz) and return {bucket: float}.
    Returns None if PANNs is unavailable.
    PANNs expects 32 kHz input.
    """
    _load_panns()
    if _panns_model == "UNAVAILABLE" or _panns_model is None:
        return None
    try:
        import librosa as _lr
        # PANNs expects 32 kHz
        y32 = _lr.resample(y, orig_sr=YAMNET_SR, target_sr=32000)
        audio_in = y32[None, :]          # (1, samples)
        clipwise, _ = _panns_model.inference(audio_in)
        scores = clipwise[0]             # (527,)
        return _scores_to_bucket_votes(_panns_labels, scores.tolist())
    except Exception as exc:
        print(f"[Ensemble] PANNs inference error: {exc}")
        return None


def _spectral_bucket_scores(feats):
    """
    Build a {bucket: float} dict from spectral + F0 features.
    These are hand-crafted confidence estimates, not neural outputs.
    Values should be in [0, 1] range.
    """
    scores  = {b: 0.0 for b in _ALL_BUCKETS}
    centroid    = feats["centroid"]
    zcr         = feats["zcr"]
    flatness    = feats["flatness"]
    onset_rate  = feats["onset_rate"]
    attack_time = feats["attack_time"]
    sub_ratio   = feats["sub_ratio"]
    bass_ratio  = feats["bass_ratio"]
    mid_ratio   = feats["mid_ratio"]
    pres_ratio  = feats["presence_ratio"]
    air_ratio   = feats["air_ratio"]
    f0          = feats["f0_median"]
    f0_conf     = feats["f0_confidence"]
    harm        = feats["harmonic_ratio"]
    rms         = feats["rms"]

    # Silence
    if rms < 5e-4 or centroid < 1.0:
        scores["SILENCE"] = 1.0
        return scores

    # ── Percussion confidence
    is_perc = onset_rate > 6 or (attack_time < 0.015 and zcr > 0.05 and harm < 0.5)
    if is_perc:
        scores["KICK_BASS"]    = max(0.0, (sub_ratio + bass_ratio) - pres_ratio - air_ratio
                                    + (0.15 if centroid < 300 else 0)
                                    + (0.10 if zcr < 0.07 else 0) - (0.2 if harm > 0.6 else 0))
        scores["SNARE_CLAP"]   = max(0.0, mid_ratio
                                    + (0.15 if 1200 < centroid < 6000 else 0)
                                    + (0.15 if zcr > 0.08 else 0)
                                    + (0.10 if 0.2 < flatness < 0.7 else 0))
        scores["HIHAT_CYMBAL"] = max(0.0, pres_ratio + air_ratio
                                    + (0.2 if centroid > 5000 else 0)
                                    + (0.2 if zcr > 0.12 else 0))
        scores["DRUM_MISC"]    = 0.05

    # ── Pitched-instrument confidence
    #
    # KEY PHYSICAL INSIGHT — Bass vs Flute/Wind:
    #   Bass guitar body resonance ALWAYS generates energy below 250 Hz, even
    #   at high fret positions (200-300 Hz notes).  A flute at the same pitch
    #   produces virtually ZERO energy below 250 Hz.
    #   → (sub_ratio + bass_ratio) is the definitive Bass vs Wind gate.
    #
    low_band = sub_ratio + bass_ratio   # energy below ~250 Hz

    if f0 is not None and f0_conf > 0.3:

        # BASS: extended F0 range (up to 350 Hz for high fret positions),
        # but MUST have significant low-band energy from body resonance.
        if f0 < 350 and low_band > 0.20:
            scores["BASS"] = min(1.0, 0.55 + 0.40 * f0_conf * harm
                                 + (0.10 if low_band > 0.45 else 0))

        # WIND / FLUTE: tonal, sustained, and — crucially — NO low-band energy.
        wind_f0_ok     = 250 < f0 < 2100
        wind_tonal     = harm > 0.60 and flatness < 0.09
        wind_sustained = onset_rate < 5.0
        wind_no_lows   = low_band < 0.15   # hard gate: flute has no sub/bass content
        if wind_f0_ok and wind_tonal and wind_sustained and wind_no_lows:
            scores["WIND"] = min(1.0, 0.60 + 0.30 * f0_conf)

        # VOCAL: mid F0, not bass-dominant
        if 100 < f0 < 900 and low_band < 0.45:
            scores["VOCAL"] = min(1.0, 0.40 + 0.30 * f0_conf
                                  + (0.20 if mid_ratio > 0.30 else 0))

        # Conflict resolution: if both BASS and WIND scored, energy decides
        if scores["BASS"] > 0 and scores["WIND"] > 0:
            if low_band > 0.20:
                scores["WIND"] = 0.0   # has low-end → bass, not wind
            else:
                scores["BASS"] = 0.0   # no low-end  → wind, not bass

    else:
        # No reliable F0 — use band energy and timbre
        if low_band > 0.45 and centroid < 700 and harm > 0.35:
            scores["BASS"]  = low_band
        if mid_ratio > 0.35 and centroid > 600 and harm > 0.45:
            scores["VOCAL"] = mid_ratio * harm
        # Wind without F0: very tonal, no low-end, sustained
        if low_band < 0.12 and harm > 0.65 and flatness < 0.08 and onset_rate < 4:
            scores["WIND"]  = harm * (1.0 - low_band)

    # Guitar: broad harmonics, plucked timbre
    if not is_perc:
        gtr = ((bass_ratio + mid_ratio) * 0.6
               + (0.2 if 0.05 < flatness < 0.50 else 0)
               + (0.1 if harm > 0.4 else 0)
               + (0.1 if 70 < (f0 or 0) < 1400 else 0))
        scores["GUITAR"] = max(0.0, min(gtr, 1.0))

        keys = (mid_ratio * 0.5
                + (0.2 if flatness < 0.25 else 0)
                + (0.15 if 300 < centroid < 4000 else 0)
                + (0.1 if harm > 0.5 else 0))
        scores["KEYS"] = max(0.0, min(keys, 1.0))

        synth = ((mid_ratio + pres_ratio) * 0.5
                 + (0.1 if flatness > 0.4 else 0))
        scores["SYNTH"] = max(0.0, min(synth, 1.0))

    return scores


def _ensemble_vote(yamnet_scores_raw, feats, y):
    """
    Combine YAMNet, PANNs (optional), and spectral votes with weights.

    Returns (winning_bucket, score_dict, model_contributions_str)
    """
    yamnet_votes   = _yamnet_bucket_votes(yamnet_scores_raw)
    spectral_votes = _spectral_bucket_scores(feats)
    panns_votes    = _panns_bucket_votes(y)   # None if PANNs unavailable

    # Normalise each vote dict so rich models don't dominate just from scale
    def _normalise(d):
        total = sum(d.values()) + 1e-9
        return {k: v / total for k, v in d.items()}

    yamnet_n   = _normalise(yamnet_votes)
    spectral_n = _normalise(spectral_votes)

    # Rebalance weights when PANNs is absent
    if panns_votes is not None:
        panns_n  = _normalise(panns_votes)
        w_y, w_p, w_s = WEIGHT_YAMNET, WEIGHT_PANNS, WEIGHT_SPECTRAL
        models_used = "YAMNet + PANNs + spectral"
    else:
        panns_n  = {b: 0.0 for b in _ALL_BUCKETS}
        # Redistribute PANNs weight evenly to the other two
        extra = WEIGHT_PANNS / 2.0
        w_y, w_p, w_s = WEIGHT_YAMNET + extra, 0.0, WEIGHT_SPECTRAL + extra
        models_used = "YAMNet + spectral"

    combined = {}
    for bucket in _ALL_BUCKETS:
        combined[bucket] = (
            w_y * yamnet_n.get(bucket, 0.0)
            + w_p * panns_n.get(bucket, 0.0)
            + w_s * spectral_n.get(bucket, 0.0)
        )

    winner = max(combined, key=combined.get)
    return winner, combined, models_used


def _resolve_audio_path(filepath):
    """
    If filepath points to a macOS resource-fork sidecar (._filename),
    resolve it to the real audio file.
    """
    if not filepath:
        return filepath
    dirname  = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    if basename.startswith("._"):
        real_path = os.path.join(dirname, basename[2:])
        if os.path.isfile(real_path):
            return real_path
    return filepath


def _classify_audio_yamnet(filepath):
    """
    Load the first *active* (non-silent) segment, run the full ensemble
    classifier, and return (label, top_yamnet_class_index, top_yamnet_name,
    feats, ensemble_scores).

    Classification priority:
        0. Truly silent audio  → returns "SILENCE"
        1. Filename keyword hint (most reliable, skips model inference)
        2. Ensemble vote: YAMNet + PANNs + spectral heuristics
        3. Raw top YAMNet label as last resort
    """
    _load_yamnet()
    _load_panns()   # no-op if not installed

    # ── Priority 0: bail out immediately for truly silent files
    y = _load_active_segment(filepath, sr=YAMNET_SR, target_duration=4.0)
    if y is None:
        dummy_feats = {
            "centroid": 0.0, "zcr": 0.0, "flatness": 1.0, "rms": 0.0,
            "onset_rate": 0.0, "attack_time": 0.05,
            "sub_ratio": 0.0, "bass_ratio": 0.0, "mid_ratio": 0.0,
            "presence_ratio": 0.0, "air_ratio": 0.0,
            "f0_median": None, "f0_confidence": 0.0, "harmonic_ratio": 0.0,
            "mfcc": np.zeros(13), "mfcc_delta_var": 0.0,
        }
        return "SILENCE", -1, "Silence", dummy_feats, {}

    # ── Priority 1: filename hint — if unambiguous, skip inference entirely
    filename_hint = _label_from_filename(filepath)

    # ── Extract spectral + F0 features (always needed)
    feats = _extract_features(y, YAMNET_SR)

    # ── Double-check silence via features
    if feats["rms"] < 5e-4 or feats["centroid"] < 1.0:
        return "SILENCE", -1, "Silence", feats, {}

    # ── Run YAMNet
    yamnet_scores_raw, _, _ = _yamnet_model(y)   # (frames, 521)

    # Raw top YAMNet class for logging
    scores_arr = np.array(yamnet_scores_raw)
    if scores_arr.ndim > 1:
        scores_arr = scores_arr.mean(axis=0)
    top_idx  = int(np.argmax(scores_arr))
    top_name = (
        _yamnet_class_names[top_idx]
        if top_idx < len(_yamnet_class_names)
        else f"Class_{top_idx}"
    )

    # ── Ensemble vote
    ensemble_bucket, ensemble_scores, models_used = _ensemble_vote(
        yamnet_scores_raw, feats, y
    )

    # ── Priority resolution
    if ensemble_bucket == "SILENCE":
        label = "SILENCE"
    elif filename_hint:
        label = filename_hint
        print(f"   📂 Filename hint: {filename_hint}  "
              f"(ensemble said: {ensemble_bucket})")
    else:
        label = ensemble_bucket or top_name
        print(f"   🧠 Ensemble [{models_used}]: {label}")

    return label, top_idx, top_name, feats, ensemble_scores


def analyze_and_organize(rename=True):
    import traceback
    project = reapy.Project()
    print(f"🔍 Deep Scanning {len(project.tracks)} tracks in REAPER...\n")

    for track in project.tracks:
        if len(track.items) == 0:
            print(f"⏭️  Skipping '{track.name}' (no items)")
            continue

        try:
            item     = track.items[0]
            take     = item.active_take
            filepath = take.source.filename
            filepath = _resolve_audio_path(filepath)

            print(f"🎵 Analyzing track {track.index + 1}: '{track.name}'")
            print(f"   Source path: {filepath!r}")

            if not filepath:
                print(f"   ⚠️ No source filename — skipping.\n")
                continue

            if not os.path.isfile(filepath):
                print(f"   ⚠️ File not found on disk — skipping.\n")
                continue

            label, class_idx, yamnet_top, feats, ens_scores = _classify_audio_yamnet(filepath)

            # Always cache the label for export_tracks() to use later,
            # regardless of whether we rename the REAPER track.
            _track_labels[filepath] = label

            # ── Skip silent tracks — don’t rename or recolour
            if label == "SILENCE":
                print(f"   🔇 Track is silent or empty — skipping rename/colour.\n")
                continue

            instrument_safe = _sanitize_track_name(label)
            color = _color_for_label(label)

            print(f"   YAMNet top : {yamnet_top} (class {class_idx})")
            print(f"   Bucket     : {label}")
            f0_str = (
                f"{feats['f0_median']:.1f} Hz (conf {feats['f0_confidence']:.2f})"
                if feats['f0_median'] is not None else "unvoiced"
            )
            print(f"   F0         : {f0_str}")
            print(f"   harm_ratio : {feats['harmonic_ratio']:.2f}  "
                  f"centroid={feats['centroid']:.0f} Hz  "
                  f"zcr={feats['zcr']:.3f}  "
                  f"flatness={feats['flatness']:.3f}  "
                  f"onset={feats['onset_rate']:.1f}/s")
            print(f"   bands      sub={feats['sub_ratio']:.2f}  "
                  f"bass={feats['bass_ratio']:.2f}  "
                  f"mid={feats['mid_ratio']:.2f}  "
                  f"pres={feats['presence_ratio']:.2f}  "
                  f"air={feats['air_ratio']:.2f}")
            # Top ensemble scores
            if ens_scores:
                top3 = sorted(ens_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                top3_str = "  ".join(f"{b}={s:.3f}" for b, s in top3)
                print(f"   ensemble   {top3_str}")

            if rename:
                track.name = f"{instrument_safe}_{track.index + 1}"
                print(f"   ✅ Identified as: {label} | Track renamed & coloured!\n")
            else:
                print(f"   ✅ Identified as: {label} | Track coloured (rename off).\n")
            track.color = color

        except Exception as e:
            print(f"⚠️ Error on '{track.name}': {e}")
            traceback.print_exc()

    print("🎉 Full DAW Organization Complete! Check REAPER.")


def _category_from_track(track_name, source_path):
    """
    Determine the export category for a track.
    Priority:
      1. classification cache (populated by analyze_and_organize)
      2. parse the track name (e.g. 'VOCAL_9' → 'VOCAL')
      3. fallback to 'OTHER'
    """
    # 1. Cache lookup
    if source_path and source_path in _track_labels:
        lbl = _track_labels[source_path]
        if lbl != "SILENCE":
            return lbl

    # 2. Parse track name
    safe = _sanitize_track_name(track_name).upper()
    ordered = sorted(_ALL_BUCKETS, key=len, reverse=True)
    for bucket in ordered:
        if safe.startswith(bucket):
            return bucket

    return "OTHER"


# -------------------------------------------------------
# Function Flask will call
# -------------------------------------------------------

def run_optimization(target, num_stems, basis, rename):
    print("\n⚙️ Running Optimization Pipeline...")
    print(f"Target: {target}")
    print(f"Number of Stems: {num_stems}")
    print(f"Basis: {basis}")
    print(f"Rename Tracks: {rename}\n")

    analyze_and_organize(rename)


def export_tracks(target, num_stems, basis, rename=True):
    """
    Collect all classified tracks, write each WAV into a per-category
    sub-folder inside a ZIP, and return the ZIP as raw bytes.

    ZIP layout:
        stems_export/
            VOCAL/
                VOCAL_9.wav
                VOCAL_13.wav
            BASS/
                BASS_7.wav
            GUITAR/
                GUITAR_3.wav
            ...
    """
    import io, zipfile

    project = reapy.Project()
    print("❌️ Starting zip export...")

    if target == "selected":
        tracks = project.selected_tracks
    else:
        tracks = project.tracks

    # In-memory zip buffer
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for track in tracks:
            if len(track.items) == 0:
                print(f"⏭️ Skipping {track.name} (no items)")
                continue

            item        = track.items[0]
            take        = item.active_take
            source_path = _resolve_audio_path(take.source.filename)

            if not source_path or not os.path.isfile(source_path):
                print(f"⚠️ Skipping {track.name}: source file not found")
                continue

            safe_name = _sanitize_track_name(track.name)

            # ── Category: classification cache first, then name parse
            category = _category_from_track(track.name, source_path)

            # ── Filename inside the zip:
            #    rename=True  → track was renamed to e.g. "VOCAL_9" → use that
            #    rename=False → keep the original source audio filename
            orig_stem   = os.path.splitext(os.path.basename(source_path))[0]
            export_name = safe_name if rename else orig_stem

            # Load with original sample rate, preserve stereo if present
            try:
                y, sr = librosa.load(source_path, sr=None, mono=False)
            except Exception as exc:
                print(f"⚠️ Could not load {track.name}: {exc}")
                continue

            # Encode WAV to an in-memory buffer
            wav_buf = io.BytesIO()
            sf.write(wav_buf, y.T if y.ndim > 1 else y, sr, format="WAV")
            wav_bytes = wav_buf.getvalue()

            zip_path = f"stems_export/{category}/{export_name}.wav"
            zf.writestr(zip_path, wav_bytes)
            print(f"  ✅  {track.name} → {zip_path}")

    zip_buffer.seek(0)
    print("✅ Zip export complete!")
    return zip_buffer