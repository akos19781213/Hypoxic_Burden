import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import os
import sys

# Import your calculation logic:
import HB_python_EDF
import numpy as np
import matplotlib.pyplot as plt


class HypoxicBurdenApp:
    def __init__(self, master):
        self.master = master
        master.title("Hypoxic Burden Calculator")

        # File selection
        self.infile_path = tk.StringVar()

        tk.Label(master, text="Input file (.edf):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.infile_entry = tk.Entry(master, textvariable=self.infile_path, width=50)
        self.infile_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(master, text="Browse...", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)

        # Start button
        self.start_btn = tk.Button(master, text="Start Calculation", command=self.start_calculation)
        self.start_btn.grid(row=1, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        # Console panel
        tk.Label(master, text="Console Output:").grid(row=2, column=0, sticky='w', padx=5)
        self.console = scrolledtext.ScrolledText(master, width=70, height=16, state='disabled', font=("Consolas", 10))
        self.console.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("EDF files", "*.edf"), ("XML files", "*.xml"), ("All Files", "*.*")]
        )
        if file_path:
            self.infile_path.set(file_path)

    def start_calculation(self):
        infile = self.infile_path.get()
        if not infile or not os.path.isfile(infile):
            messagebox.showerror("Error", "Please select a valid input file.")
            return
        # Run calculation in a separate thread to avoid freezing UI
        threading.Thread(target=self.run_calculation, args=(infile,), daemon=True).start()

    def run_calculation(self, infile):
        self.clear_console()
        self.write_console("Hypoxic Burden Calculator\n------------------------\n")
        ext = os.path.splitext(infile)[-1].lower()
        try:
            if ext == '.edf':
                self.write_console(f"Loading ApneaLink EDF+ file: {infile}\n")
                spo2, events, stage = HB_python_EDF.import_apnealink_edf(infile)
            elif ext == '.xml':
                self.write_console("XML workflow requires SpO2 data from CSV. Not implemented in this app.\n")
                return
            else:
                self.write_console("Unsupported file type! Please provide an EDF+ (.edf) file.\n")
                return

            self.write_console(f"SpO2 length: {len(spo2.sig)}, SR: {spo2.sr}\n")
            from collections import Counter
            event_counts = Counter(events.type)
            self.write_console("Event counts by type:\n")
            for event_type, count in event_counts.items():
                self.write_console(f"{event_type}: {count}\n")
            hb_value = HB_python_EDF.calc_hb(spo2, events, stage)
            hb_value, hours, minutes, hours_sleep = HB_python_EDF.calc_hb(spo2, events, stage)
            self.write_console(f"Hours sleep: {hours}h {minutes}m ({hours_sleep:.2f} hours)\n")
            self.write_console(f"\nHypoxic Burden value: {hb_value:.4f}\n")
            
        except Exception as e:
            self.write_console(f"\nError calculating Hypoxic Burden: {e}\n")

    def write_console(self, text):
        self.console.configure(state='normal')
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.console.configure(state='disabled')

    def clear_console(self):
        self.console.configure(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.configure(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = HypoxicBurdenApp(root)
    root.mainloop()
