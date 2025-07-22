import os
import math
import csv
import warnings
import numpy as np
import xml.etree.ElementTree as ET
from scipy.signal import filtfilt, find_peaks, remez
from tqdm import tqdm

# ──────────────────────────────────────────
# Lightweight stand‑in "struct" classes
# ──────────────────────────────────────────
class SpO2Struct:
    """Container for SpO2 signal and sample rate."""

    def __init__(self, sig, sr=1):
        self.sig = sig
        self.sr = sr


class RespEventsStruct:
    """Container for respiratory event annotations."""

    def __init__(self, ev_type, start, duration):
        self.type = ev_type
        self.start = np.asarray(start)
        self.duration = np.asarray(duration)


class SleepStageStruct:
    """Container for sleep‑stage hypnogram (per‑second, integer codes)."""

    def __init__(self, annotation, sr=1):
        self.annotation = annotation
        self.sr = sr


# ──────────────────────────────────────────
# calc_hb  — Python port of MATLAB calcHB v1.1
# ──────────────────────────────────────────

def calc_hb(spo2, events, stage):
    """Compute Hypoxic Burden (%·min per hr) for a single night.

    Parameters
    ----------
    spo2 : SpO2Struct
        Oxygen saturation trace sampled at 1 Hz.
    events : RespEventsStruct
        Apnoea/hypopnoea events (start time & duration in seconds).
    stage : SleepStageStruct
        Per‑second sleep stage codes (0 = wake, 1/2/3 = NREM, 5 = REM, 9 = indeterminate).
    """

    # Sample‑rate check (assume 1 Hz throughout)
    if spo2.sr != 1 or stage.sr != 1:
        raise ValueError("SpO2 and sleep‑stage sample rates must both be 1 Hz.")

    if len(events.type) == 0:
        return float("nan")

    # Replace physiologically impossible SpO2 values with NaN
    spo2_phys = spo2.sig.astype(float)
    spo2_phys[(spo2_phys < 50) | (spo2_phys > 100)] = np.nan

    num_events = len(events.type)
    sig_len = len(spo2_phys)

    # Mean event duration & inter‑event gap
    dur_avg = int(math.ceil(float(np.nanmean(events.duration))))

    if len(events.start) > 1:
        gap_avg = int(math.ceil(float(np.nanmean(np.diff(events.start)))))
    else:
        gap_avg = 60  # fallback gap (s)

    # Event‑aligned average
    win_pts = 240 * spo2.sr + 1        # −120 s … +120 s window
    evt_mat = np.full((win_pts, num_events), np.nan)

    for k in range(num_events):
        finish = events.start[k] + events.duration[k]
        finish_idx = int(round(finish * spo2.sr))
        beg = finish_idx - 120 * spo2.sr
        end = finish_idx + 120 * spo2.sr
        if beg >= 0 and end < sig_len:
            evt_mat[:, k] = spo2_phys[beg:end + 1]

    spo2_avg = np.nanmean(evt_mat, axis=1)

    # 0.03 Hz low‑pass FIR filter
    if spo2.sr == 1:
        B = np.array([
            1.09398212241e-04, 5.14594526374e-04, 1.35039717994e-03,
            2.34170006253e-03, 2.48594032701e-03, 2.07543145171e-04,
           -5.65945034423e-03,-1.42580878081e-02,-2.14154813834e-02,
           -1.99694177499e-02,-2.42512010346e-03, 3.47944528214e-02,
            8.76956913669e-02, 1.44171828096e-01, 1.87717212245e-01,
            2.04101948813e-01, 1.87717212245e-01, 1.44171828096e-01,
            8.76956913669e-02, 3.47944528214e-02,-2.42512010346e-03,
           -1.99694177499e-02,-2.14154813834e-02,-1.42580878081e-02,
           -5.65945034423e-03, 2.07543145171e-04, 2.48594032701e-03,
            2.34170006253e-03, 1.35039717994e-03, 5.14594526374e-04,
            1.09398212241e-04
        ])
    else:
        filt_len = 30 * spo2.sr
        B = remez(filt_len + 1, [0, 1/30, 1/15, 0.5], [1, 0], Hz=spo2.sr)
    spo2_filt = filtfilt(B, [1.0], spo2_avg)

    # Crop average response window
    s_idx = 120 * spo2.sr + 1 - dur_avg * spo2.sr
    e_idx = 120 * spo2.sr + 1 + min(90, gap_avg) * spo2.sr
    time_zero = dur_avg
    spo2_resp = spo2_filt[s_idx:e_idx + 1]

    # Identify nadir & bounding peaks
    nadir_idx = None
    win_start = None
    win_end = None

    neg_peaks, neg_idx = find_peaks(-spo2_resp)
    if len(neg_idx) > 0:
        big_peak_pos = int(np.argmax(neg_peaks))
        nadir_idx = neg_idx[big_peak_pos]
        nadir_val = -neg_peaks[big_peak_pos]

        # Left (onset) peak
        pk_left, idx_left = find_peaks(spo2_resp[:nadir_idx])
        if len(idx_left) > 0:
            max_on = pk_left.max()
            ok = idx_left[pk_left - nadir_val > 0.75 * (max_on - nadir_val)]
            if len(ok) > 0:
                win_start = ok[-1]

        # Right (offset) peak
        pk_right, idx_right = find_peaks(spo2_resp[nadir_idx:])
        if len(idx_right) > 0:
            max_off = pk_right.max()
            ok = idx_right[pk_right - nadir_val > 0.75 * (max_off - nadir_val)]
            if len(ok) > 0:
                win_end = ok[0] + nadir_idx

    if nadir_idx is None or win_start is None or win_end is None:
        warnings.warn("Window search failed — falling back to default limits")
        win_start = time_zero - 5
        win_end = time_zero + 45

    win_start -= time_zero
    win_end -= time_zero

    # Sum area (%·min) for each event
    pct_min_total = 0.0
    limit_idx = 0

    for k in range(num_events):
        finish = events.start[k] + events.duration[k]
        finish_idx = int(round(finish * spo2.sr))

        if finish_idx - 100 < 0 or finish_idx + win_end >= sig_len:
            continue

        baseline = np.nanmax(spo2_phys[finish_idx - 100:finish_idx + 1])

        seg_beg = max(finish_idx + win_start, limit_idx)
        seg_end = finish_idx + win_end

        seg = baseline - spo2_phys[seg_beg:seg_end + 1]
        seg[seg < 0] = 0
        pct_min_total += np.nansum(seg) / 60.0
        limit_idx = seg_end

    # Normalise by sleep time (h)
    sleep_mask = (stage.annotation > 0) & (stage.annotation < 9)
    hours_sleep = float(np.sum(~np.isnan(spo2_phys[sleep_mask]))) / 3600.0
    if hours_sleep == 0:
        return float("nan")
    return pct_min_total / hours_sleep


# ──────────────────────────────────────────
# XML parser (based on <ScoredEvent> blocks)
# ──────────────────────────────────────────

stage_map = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "REM sleep|5": 5
}


def parse_xml(xml_path, spo2_len):
    """Parse NSRR XML annotation and return events + sleep stages."""

    tree = ET.parse(xml_path)
    root = tree.getroot()

    ev_type, ev_start, ev_dur = [], [], []
    hyp = np.full(spo2_len, 9, dtype=np.int16)  # default = Indeterminate

    for node in root.findall('.//ScoredEvent'):
        label = node.findtext('EventConcept', '')
        start = float(node.findtext('Start', '0'))
        duration = float(node.findtext('Duration', '0'))

        # Sleep stage
        if label in stage_map:
            code = stage_map[label]
            beg = int(start)
            end = int(start + duration)
            end = min(end, spo2_len)
            hyp[beg:end] = code
            continue

        # AHI‑related events
        low = label.lower()
        if 'apnea' in low and 'hypopnea' not in low:
            ev_type.append('Apnea')
        elif 'hypopnea' in low:
            ev_type.append('Hypopnea')
        elif 'unsure' in low:
            ev_type.append
