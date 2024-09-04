import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import openpyxl

# File path and sheet details
file_path = os.path.join(os.getcwd(), 'SiEPIC openEBL Process Control Monitoring (PCM) Report.xlsx')
sheet_name = 'ANT'

# Define which measurement type, wavelength, and polarization to plot
measurement_type_to_plot = 'Y-Branch Loss (dB)' # 'Bragg Gratings Wavelength Drift (nm)' # 'Straight Waveguides (dB/cm)'
wavelength_to_plot = 1550
polarization_to_plot = 'TE'

def extract_alt_id(file_path, sheet_name):
    df_name = pd.read_excel(file_path, sheet_name=sheet_name)

    # Extract Alt. ID from the second row, starting from the fourth column
    alt_id_list = df_name.iloc[0, 3:].astype(str).values.tolist()
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=5)

    return df, alt_id_list

def split_value_uncertainty(cell_value):
    if isinstance(cell_value, str):
        if '±' in cell_value:
            match = re.match(r'([0-9]*\.?[0-9]+)\s*±\s*([0-9]*\.?[0-9]+)', cell_value)
            if match:
                return float(match.group(1)), float(match.group(2))
        elif 'nm' in cell_value:
            # Handle cases where the value is followed by 'nm' without uncertainty
            try:
                value_str = re.sub(r'\s*nm', '', cell_value)
                return float(value_str), np.nan
            except ValueError:
                pass
    return np.nan, np.nan  # Return NaN for both value and uncertainty if parsing fails

def plot_measurement_type(df, alt_id_list, measurement_type, wavelength, polarization):
    # Filter the DataFrame based on the measurement type, wavelength, and polarization
    df_filtered = df[(df['Type'] == measurement_type) &
                     (df['Wavelength (nm)'].astype(str) == str(wavelength)) &
                     (df['Polarization'] == polarization)]

    # Check if there's any data to plot
    if df_filtered.empty:
        print("No data to plot after filtering.")
        return

    values = []
    uncertainties = []
    valid_alt_ids = []

    for i, sample in enumerate(df_filtered.columns[3:]):  # All columns after the first three
        cell_value = df_filtered[sample].values[0]
        value, uncertainty = split_value_uncertainty(cell_value)

        if not np.isnan(value):
            values.append(value)
            uncertainties.append(uncertainty)
            valid_alt_ids.append(alt_id_list[i])

    if not values:
        print("No valid numeric data to plot.")
        return

    plt.figure(figsize=(12, 6))

    # Plot the valid data
    plt.errorbar(valid_alt_ids, values, yerr=uncertainties, fmt='o', capsize=5)

    plt.xlabel('Alt. ID')
    plt.ylabel(f'{measurement_type.split("(")[0]} (nm)')
    plt.title(f'{measurement_type} at {wavelength} nm ({polarization} Polarization)')
    plt.xticks(rotation=45, ha="right")  # Rotate the x-axis labels for better readability

    # Display the plot
    plt.tight_layout()
    plt.show()

# Usage
df, alt_id_list = extract_alt_id(file_path, sheet_name)

# Plot the selected data
plot_measurement_type(df, alt_id_list, measurement_type_to_plot, wavelength_to_plot, polarization_to_plot)