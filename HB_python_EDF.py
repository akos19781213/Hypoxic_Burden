import os
import math
import warnings
import numpy as np
import xml.etree.ElementTree as ET
import sys
import argparse
from scipy.signal import filtfilt, find_peaks, remez, resample
from tqdm import tqdm

# Use mne for tolerant EDF reading
import mne

# ──────────────────────────────────────────
# A simple class serving as a substitute for a structure for data storage
# ──────────────────────────────────────────
class SpO2Struct:
    def __init__(self, sig, sr=1):
        self.sig = sig
        self.sr = sr

class RespEventsStruct:
    def __init__(self, ev_type, start, duration):
        self.type = ev_type
        self.start = np.asarray(start)
        self.duration = np.asarray(duration)

class SleepStageStruct:
    def __init__(self, annotation, sr=1):
        self.annotation = annotation
        self.sr = sr

# ──────────────────────────────────────────
# calc_hb  (MATLAB calcHB v1.1 Porting)
# ──────────────────────────────────────────
def calc_hb(spo2, events, stage):
    if spo2.sr != 1 or stage.sr != 1:
        raise ValueError("The SpO₂ and SleepStage sample rate must be 1 Hz.")

    if len(events.type) == 0:
        return float('nan')

    spo2_phys = spo2.sig.astype(float)
    spo2_phys[(spo2_phys < 50) | (spo2_phys > 100)] = np.nan

    num_events = len(events.type)
    sig_len = len(spo2_phys)
    dur_avg = int(math.ceil(float(np.nanmean(events.duration))))

    if len(events.start) > 1:
        gap_avg = int(math.ceil(float(np.nanmean(np.diff(events.start)))))
    else:
        gap_avg = 60

    win_pts = 240 * spo2.sr + 1
    evt_mat = np.full((win_pts, num_events), np.nan)

    for k in range(num_events):
        finish = events.start[k] + events.duration[k]
        finish_idx = int(round(finish * spo2.sr))
        beg = finish_idx - 120 * spo2.sr
        end = finish_idx + 120 * spo2.sr
        if beg >= 0 and end < sig_len:
            evt_mat[:, k] = spo2_phys[beg:end + 1]

    spo2_avg = np.nanmean(evt_mat, axis=1)

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

    s_idx = 120 * spo2.sr + 1 - dur_avg * spo2.sr
    e_idx = 120 * spo2.sr + 1 + min(90, gap_avg) * spo2.sr
    time_zero = dur_avg
    spo2_resp = spo2_filt[s_idx:e_idx + 1]

    nadir_idx = None
    win_start = None
    win_end = None

    neg_peaks, neg_idx = find_peaks(-spo2_resp)
    if len(neg_idx) > 0:
        big_peak_pos = int(np.argmax(neg_peaks))
        nadir_idx = neg_idx[big_peak_pos]
        nadir_val = -neg_peaks[big_peak_pos]

        pk_left, idx_left = find_peaks(spo2_resp[:nadir_idx])
        if len(idx_left) > 0:
            max_on = pk_left.max()
            ok = idx_left[pk_left - nadir_val > 0.75 * (max_on - nadir_val)]
            if len(ok) > 0:
                win_start = ok[-1]

        pk_right, idx_right = find_peaks(spo2_resp[nadir_idx:])
        if len(idx_right) > 0:
            max_off = pk_right.max()
            ok = idx_right[pk_right - nadir_val > 0.75 * (max_off - nadir_val)]
            if len(ok) > 0:
                win_end = ok[0] + nadir_idx

    if nadir_idx is None or win_start is None or win_end is None:
        warnings.warn("Window navigation failed, using default value")
        win_start = time_zero - 5
        win_end = time_zero + 45

    win_start -= time_zero
    win_end -= time_zero

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

    sleep_mask = (stage.annotation > 0) & (stage.annotation < 9)
    hours_sleep = float(np.sum(~np.isnan(spo2_phys[sleep_mask]))) / 3600.0
	# Convert hours_sleep to hours and minutes format
    hours = int(hours_sleep)
    minutes = int(round((hours_sleep - hours) * 60))
    print(f"Hours sleep: {hours}h {minutes}m ({hours_sleep:.2f} hours)")
    if hours_sleep == 0:
        return float('nan')
    return pct_min_total / hours_sleep

# ──────────────────────────────────────────
# XML Parser (ScoredEvent-based)
# ──────────────────────────────────────────
stage_map = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "REM sleep|5": 5
}

def parse_xml(xml_path, spo2_len):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ev_type, ev_start, ev_dur = [], [], []
    hyp = np.full(spo2_len, 9, dtype=np.int16)  # Default

    for node in root.findall('.//ScoredEvent'):
        label = node.findtext('EventConcept', '')
        start = float(node.findtext('Start', '0'))
        duration = float(node.findtext('Duration', '0'))

        if label in stage_map:
            code = stage_map[label]
            beg = int(start)
            end = int(start + duration)
            end = min(end, spo2_len)
            hyp[beg:end] = code
            continue

        low = label.lower()
        if 'apnea' in low and 'hypopnea' not in low:
            ev_type.append('Apnea')
        elif 'hypopnea' in low:
            ev_type.append('Hypopnea')
        elif 'unsure' in low:
            ev_type.append('Unsure')
        else:
            continue
        ev_start.append(start)
        ev_dur.append(duration)

    events = RespEventsStruct(ev_type, ev_start, ev_dur)
    stage = SleepStageStruct(hyp)
    return events, stage

# ──────────────────────────────────────────
# ApneaLink EDF+ Parser (MNE-Python based)
# ──────────────────────────────────────────
def import_apnealink_edf(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='error')
    # Find SpO2 channel
    spo2_idx = None
    for i, ch in enumerate(raw.ch_names):
        if ch.startswith("Saturation"):
            spo2_idx = i
            break
    if spo2_idx is None:
        raise ValueError("Could not find SpO2 channel named 'Saturation' in EDF file.")

    spo2 = raw.get_data(picks=[spo2_idx]).flatten()
    sr = int(raw.info['sfreq'])
    
    # Resample to 1 Hz
    if sr != 1:
        target_length = int(len(spo2) / sr)
        spo2 = resample(spo2, target_length)
        sr = 1
        
    # Read annotations/events
    events_type = []
    events_start = []
    events_duration = []
    for ann in raw.annotations:
        desc = ann['description'].lower()
        if 'hypopnea' in desc:
            events_type.append('Hypopnea')
        elif 'obstructi' in desc:
            events_type.append('Obstructive')
        elif 'unclassif' in desc:
            events_type.append('Unclassified')
        elif 'central a' in desc:
            events_type.append('Central')
        else:
            continue
        events_start.append(float(ann['onset']))
        events_duration.append(float(ann['duration']))

    hyp_dummy = np.full(len(spo2), 2, dtype=np.int16)
    sleep_stage_struct = SleepStageStruct(hyp_dummy, sr=sr)

    spo2_struct = SpO2Struct(sig=spo2, sr=sr)
    events_struct = RespEventsStruct(events_type, events_start, events_duration)
    return spo2_struct, events_struct, sleep_stage_struct

# ──────────────────────────────────────────
# Main Application Logic
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hypoxic Burden Calculator (ApneaLink EDF+ or XML)")
    parser.add_argument("infile", help="Input EDF+ file (.edf) or XML file (.xml)")
    parser.add_argument("--xml", help="Optional XML event file (for XML+CSV workflow)", default=None)
    args = parser.parse_args()

    infile = args.infile
    ext = os.path.splitext(infile)[-1].lower()

    print("\nHypoxic Burden Calculator\n------------------------")
    if ext == '.edf':
        print(f"Loading ApneaLink EDF+ file: {infile}")
        spo2, events, stage = import_apnealink_edf(infile)
    elif ext == '.xml':
        print(f"Loading XML file: {infile}")
        # Need to load SpO2 from CSV (not covered here)
        print("XML workflow requires SpO2 data from CSV. Not implemented in this app.")
        sys.exit(1)
    else:
        print("Unsupported file type! Please provide an EDF+ (.edf) file.")
        sys.exit(1)

    try:
        print(f"SpO2 length: {len(spo2.sig)}, SR: {spo2.sr}")
        print(f"Events: {events.type}, Start: {events.start}, Duration: {events.duration}")
        hb_value = calc_hb(spo2, events, stage)
        print(f"\nHypoxic Burden value: {hb_value:.4f}")
    except Exception as e:
        print(f"\nError calculating Hypoxic Burden: {e}")

if __name__ == "__main__":
    main()
