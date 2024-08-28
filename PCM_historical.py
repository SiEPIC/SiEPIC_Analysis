import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import openpyxl

class HistAnalysis:
    def __init__(self, file_path, sheet_name, measurement_type, wavelength, polarization):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.measurement_type = measurement_type
        self.wavelength = wavelength
        self.polarization = polarization
        self.df, self.alt_id_list = self.extract_alt_id()

    def extract_alt_id(self):
        df_name = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        alt_id_list = df_name.iloc[0, 3:].astype(str).values.tolist()
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, skiprows=5)
        return df, alt_id_list

    @staticmethod
    def split_value_uncertainty(cell_value):
        if isinstance(cell_value, str):
            if '±' in cell_value:
                match = re.match(r'([0-9]*\.?[0-9]+)\s*±\s*([0-9]*\.?[0-9]+)', cell_value)
                if match:
                    return float(match.group(1)), float(match.group(2))
            elif 'nm' in cell_value:
                try:
                    value_str = re.sub(r'\s*nm', '', cell_value)
                    return float(value_str), np.nan
                except ValueError:
                    pass
        return np.nan, np.nan

    def plot_measurement_type(self):
        df_filtered = self.df[(self.df['Type'] == self.measurement_type) &
                              (self.df['Wavelength (nm)'].astype(str) == str(self.wavelength)) &
                              (self.df['Polarization'] == self.polarization)]

        if df_filtered.empty:
            print("No data to plot after filtering.")
            return

        values = []
        uncertainties = []
        valid_alt_ids = []

        for i, sample in enumerate(df_filtered.columns[3:]):
            cell_value = df_filtered[sample].values[0]
            value, uncertainty = self.split_value_uncertainty(cell_value)

            if not np.isnan(value):
                values.append(value)
                uncertainties.append(uncertainty)
                valid_alt_ids.append(self.alt_id_list[i])

        if not values:
            print("No valid numeric data to plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.errorbar(valid_alt_ids, values, yerr=uncertainties, fmt='o', capsize=5)
        plt.xlabel('Alt. ID')
        plt.ylabel(f'{self.measurement_type.split("(")[0]} (nm)')
        plt.title(f'{self.measurement_type} at {self.wavelength} nm ({self.polarization} Polarization)')
        plt.xticks(rotation=45, ha="right")

        # Generate the save path
        measurement_name = re.sub(r'\s*\(.*?\)', '', self.measurement_type)
        save_path = f'{self.wavelength}{self.polarization}_{measurement_name} plot.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

# Define usage details
file_path = os.path.join(os.getcwd(), 'SiEPIC openEBL Process Control Monitoring (PCM) Report.xlsx')
sheet_name = 'ANT'
measurement_type = 'Y-Branch Loss (dB)'
wavelength = 1550
polarization = 'TE'

analysis = HistAnalysis(file_path, sheet_name, measurement_type, wavelength, polarization)
analysis.plot_measurement_type()
