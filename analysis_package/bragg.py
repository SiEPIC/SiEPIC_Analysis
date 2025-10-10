
"""
SiEPIC Analysis Package

Author:     Mustafa Hammood
            Mustafa@siepic.com

            Tenna Yuan
            tenna@student.ubc.ca

Example:    Application of SiEPIC_AP analysis functions
            Process data of various contra-directional couplers (CDCs)
            Extract the period and bandwidth from a set of devices
"""

import os
import sys
import io
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import siepic_analysis_package as siap

import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class DirectionalCoupler:
    def __init__(self, fname_data, device_prefix, port_thru, port_drop, device_suffix,
                 name, wavl, pol, main_script_directory,
                 threshold, x_min, x_max):
        self.fname_data = fname_data
        self.device_prefix = device_prefix
        self.port_thru = port_thru
        self.port_drop = port_drop
        self.device_suffix = device_suffix
        self.name = name
        self.wavl = wavl
        self.pol = pol
        self.main_script_directory = main_script_directory
        self.threshold = threshold
        self.x_min = x_min
        self.x_max = x_max

        self.devices = []
        self.period = []
        self.WL = []
        self.BW = []
        self.df_figures = pd.DataFrame()

    def getDeviceParameter(self, device_id):
        """
        Extract the waveguide length from a device ID by removing specified prefixes and suffixes.

        Args:
        device_id (str): The device ID from which to extract the waveguide length.

        Returns:
        float: The waveguide length extracted from the device ID.
        """
        try:
            start_index = device_id.index(self.device_prefix) + len(self.device_prefix)

            if self.device_suffix:  # If suffix is not empty
                end_index = device_id.index(self.device_suffix, start_index)
                value_str = device_id[start_index:end_index]
            else:
                # If suffix is empty, extract all digits and optionally decimal points
                value_str = re.findall(r'[0-9.]+', device_id[start_index:])[0]

            device_value = float(value_str)
            return device_value
        except (ValueError, IndexError):
            # Handle cases where the prefix, suffix, or device_value is not found
            return None

    def process_files(self):
        for root, dirs, files in os.walk(self.fname_data):
            if os.path.basename(root).startswith(self.device_prefix):
                for file in files:
                    if file.endswith(".csv"):
                        file_path = os.path.join(root, file)
                        device = siap.analysis.processCSV(file_path)

                        # device.dropCalib, device.ThruEnvelope, x, y = siap.analysis.calibrate_envelope(
                        #     device.wavl, device.pwr[self.port_thru], device.pwr[self.port_drop],
                        #     N_seg=self.N_seg, tol=self.tol, verbose=False)

                        if self.wavl == 1550:
                            device.dropCalib, x, y = self.bragg_calibrate(device.wavl, device.pwr[self.port_thru],
                                                                       x_min=self.x_min, x_max=self.x_max, verbose=False)

                        elif self.wavl == 1310:
                            device.dropCalib, x, y = self.bragg_calibrate(device.wavl, siap.core.smooth(device.wavl, device.pwr[self.port_thru], window=121),
                                                                      x_min=self.x_min, x_max=self.x_max, verbose=False)

                        # plt.show()

                        [device.BW, device.WL] = siap.analysis.bandwidth(device.wavl, -device.dropCalib, threshold=6)

                        self.devices.append(device)
                        self.period.append(self.getDeviceParameter(device.deviceID))
                        self.WL.append(device.WL)
                        self.BW.append(device.BW)

        return self.devices, self.period, self.WL, self.BW

    def plot_devices(self, bragg_type):
        if bragg_type == 'sweep':
            data_name = 'dW'
        else:
            data_name = 'Period'

        # Sort devices by the parameter returned by getDeviceParameter
        sorted_devices = sorted(self.devices, key=lambda device: self.getDeviceParameter(device.deviceID))

        # Raw measurement plot
        plt.figure(figsize=(10, 6))
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Serif', 'font.weight': 'bold'})
        for device in sorted_devices:
            label = data_name + ' = ' + str(self.getDeviceParameter(device.deviceID)) + ' nm'
            plt.plot(device.wavl, device.pwr[self.port_drop], label=label)

        plt.legend(loc=0)
        plt.ylabel('Power [dBm]', color='black')
        plt.xlabel('Wavelength [nm]', color='black')
        plt.title("Raw measurement of all structures")

        # Save plots
        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_devices_raw, format='pdf')
        # plt.show()  # Display graph

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_raw', 'Figure': img_buffer},
            ignore_index=True
        )

        # Calibrated measurement plot
        plt.figure(figsize=(10, 6))
        for device in sorted_devices:
            label = data_name + ' = ' + str(self.getDeviceParameter(device.deviceID)) + ' nm'
            plt.plot(device.wavl, device.dropCalib, label=label)

        plt.legend(loc=0)
        plt.ylabel('Transmission (dB)', color='black')
        plt.xlabel('Wavelength (nm)', color='black')
        plt.title("Calibrated measurement of all structures (using envelope calibration)")

        # Save plots
        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_devices_calib, format='pdf')
        # plt.show()  # Display graph

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_calib', 'Figure': img_buffer},
            ignore_index=True
        )

    def plot_analysis_results(self, bragg_type):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.scatter(self.period, self.WL, color='blue')
        if bragg_type == 'sweep':
            ax1.set_xlabel('dW [nm]')
        else:
            ax1.set_xlabel('Grating period [nm]')
        ax1.set_ylabel('Bragg wavelength [nm]', color='blue')
        ax1.tick_params(axis='y', colors='blue')

        ax2 = ax1.twinx()
        ax2.scatter(self.period, self.BW, color='red')
        ax2.set_ylabel('3 dB Bandwidth [nm]', color='red')
        ax2.tick_params(axis='y', colors='red')

        plt.title("Extracted bandwidth and central wavelength of the Bragg gratings")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Serif', 'font.weight': 'bold'})

        # save plots
        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_analysis, format='pdf')
        # plt.show()  # Display graph

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_central', 'Figure': img_buffer},
            ignore_index=True
        )

    def overlay_simulation_data(self, target_wavelength, sim_label):
        # 1550nm simulation results (220 nm SOI, air clad)
        period_sim_air = [313, 315, 317, 319, 321, 323, 324, 325, 326]
        wavl_sim_air = [1517, 1522, 1527, 1532, 1538, 1543, 1544.74, 1548.9, 1549.85]

        # 1550nm simulation results (220 nm SOI, SiO2 clad)
        simulation_period_sio2 = [313, 315, 317, 319, 321, 323]
        simulation_wavl_sio2 = [1536.64, 1542.24, 1547.85, 1553.45, 1559.06, 1564.56]

        # 1310nm simulation results (220 nm SOI, SiO2 clad)
        simulation_period_sio2_1310= [273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283]
        simulation_wavl_sio2_1310= [1318.45, 1321.05, 1323.65, 1326.26, 1328.86, 1331.46, 1334.06, 1336.54, 1339.14, 1341.7, 1344.21]

        # 1310nm simulation results (220 nm SOI, air)
        period_sim_air_1310 = [273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283]
        wavl_sim_air_1310 = [1295.48, 1297.79, 1300.2, 1302.52, 1304.94, 1307.26, 1309.57, 1311.88, 1314.08, 1316.4, 1318.82]

        if sim_label == 'Simulation (1550_SiO2 Clad)':
            simulation_period = simulation_period_sio2
            simulation_wavl = simulation_wavl_sio2
        elif sim_label == 'Simulation (1550_Air Clad)':
            simulation_period = period_sim_air
            simulation_wavl = wavl_sim_air
        elif sim_label == 'Simulation (1310_SiO2 Clad)':
            simulation_period = simulation_period_sio2_1310
            simulation_wavl = simulation_wavl_sio2_1310
        elif sim_label == 'Simulation (1310_Air Clad)':
            simulation_period = period_sim_air_1310
            simulation_wavl = wavl_sim_air_1310
        else:
            simulation_period = None
            simulation_wavl = None
            print('Simulation data not specified')

        # Interpolate simulation period at target_wavelength_sim
        simulation_period_at_target_sim = np.interp(target_wavelength, simulation_wavl, simulation_period)

        # Interpolate experimental period at target_wavelength_exp
        experimental_period_at_target_exp = np.interp(target_wavelength, self.WL, self.period)

        print(f"Simulation period at {target_wavelength} nm: {simulation_period_at_target_sim} nm")
        print(f"Experimental period at {target_wavelength} nm: {experimental_period_at_target_exp} nm")

        plt.figure(figsize=(10, 6))
        plt.scatter(self.period, self.WL, color='r', marker='x', label='Experiment')
        plt.scatter(simulation_period, simulation_wavl, color='b', marker='o', label=sim_label)

        plt.legend()
        plt.ylabel('Bragg Wavelength [nm]', color='black')
        plt.xlabel('Grating Period [nm]', color='black')
        plt.title("Comparison of Bragg wavelength between simulation and experiment.")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Serif', 'font.weight': 'bold'})

        # Define a common set of wavelengths for interpolation
        common_wavelengths = np.linspace(min(min(self.period), min(simulation_period)),
                                         max(max(self.period), max(simulation_period)), 100)

        # Interpolate both experimental and simulated data to the common set of wavelengths
        exp_wavelength_interp = np.interp(common_wavelengths, self.period, self.WL)
        sim_wavelength_interp = np.interp(common_wavelengths, simulation_period, simulation_wavl)

        # Fit a polynomial to the experimental data
        exp_coefficients = np.polyfit(common_wavelengths, exp_wavelength_interp, 2)
        exp_poly_func = np.poly1d(exp_coefficients)

        # Fit a polynomial to the simulated data
        sim_coefficients = np.polyfit(common_wavelengths, sim_wavelength_interp, 2)
        sim_poly_func = np.poly1d(sim_coefficients)

        # Evaluate both fit lines at the common wavelengths
        exp_wavelength_fit = exp_poly_func(common_wavelengths)
        sim_wavelength_fit = sim_poly_func(common_wavelengths)

        # Calculate the differences between the fit lines
        differences = exp_wavelength_fit - sim_wavelength_fit
        average_difference = np.mean(differences)

        print(f"Bragg Grating Wavelength Drift is: {average_difference} nm for {self.name}_{self.pol}{self.wavl}")

        # save plots
        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_analysis_WL, format='pdf')
        # plt.show()  # Display graph

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_overlay', 'Figure': img_buffer},
            ignore_index=True
        )

        return average_difference
    def saveGraph(self):
        """
        Save a graph as PDF files in a directory based on the `self.name` attribute.

        This method creates a directory named after `self.name` inside the specified `main_script_directory`
        (or a custom directory if provided) and saves two PDF files within this directory: one for raw data
        and another for cutback data.

        Returns:
        - pdf_path_raw (str): The full path to the saved raw data PDF file.
        - pdf_path_cutback (str): The full path to the saved cutback data PDF file.
        """
        # Create a directory based on self.name, self.pol, and self.wavl if it doesn't exist
        output_directory = os.path.join(self.main_script_directory, f"{self.name}_{self.pol}{self.wavl}")
        os.makedirs(output_directory, exist_ok=True)

        pdf_path_devices_raw = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_devices_raw.pdf")
        pdf_path_devices_calib = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_devices_calib.pdf")
        pdf_path_analysis = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_analysis.pdf")
        pdf_path_analysis_WL = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_analysis_WL.pdf")

        return pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL

    # functions for bragg calibration
    def bragg_deriv(self, data, threshold):
        gradient = np.gradient(data)

        # Find the positions where the gradient exceeds the threshold
        change_indices = np.where(np.abs(gradient) > threshold)[0]

        return change_indices

    def create_envelope(self, x, y):
        # Find local maxima
        peaks, _ = find_peaks(y)
        # Find local minima
        troughs, _ = find_peaks(-y)

        # Interpolate to create the envelope
        upper_envelope = np.interp(x, x[peaks], y[peaks])
        lower_envelope = np.interp(x, x[troughs], y[troughs])

        return upper_envelope, lower_envelope

    def calibrate_data(self, x, y, poly_coeffs):
        poly_fit = np.polyval(poly_coeffs, x)
        calibrated_data = y - poly_fit
        return calibrated_data

    def bragg_calibrate(self, x, y, x_min=None, x_max=None, verbose=False):
        # Ensure x and y are numpy arrays
        x = np.array(x)
        y = np.array(y)

        if verbose:
            plt.figure()
            plt.plot(x, y, label="Calibration reference")
            plt.legend(loc=0)
            plt.title("Original input data set")
            plt.xlabel("X")
            plt.ylabel("Y")

        # Smooth the data before creating the envelope
        y_smooth = siap.core.smooth(x, y, window=51, order=3)

        # Filter the data within the x range if x_min and x_max are specified
        if x_min is not None and x_max is not None:
            mask = (x >= x_min) & (x <= x_max)
            x_area = x[mask]
            y_area = y[mask]
        else:
            x_area = x
            y_area = y_smooth

        # Define a threshold for detecting sudden changes
        threshold = self.threshold # Adjust based on datasets

        # Detect sudden drops
        change_indices = self.bragg_deriv(y_area, threshold)
        change_indices = np.array(change_indices, dtype=int)

        if len(change_indices) == 0:
            print(f"No sudden drop detected in file, try adjusting the threshold.")
            return

        if x_min is not None and x_max is not None:
            # Map change indices back to the original data
            full_range_indices = np.where(mask)[0]
            mapped_change_indices = full_range_indices[change_indices]
        else:
            mapped_change_indices = change_indices

        if verbose:
            plt.figure()
            plt.plot(x, y, linewidth=0.1, label='Calibration reference')
            plt.plot(x[mapped_change_indices], y[mapped_change_indices], 'ro', label='Sudden Changes')
            plt.legend(loc=0)
            plt.title("Sampling of reference data set")
            plt.xlabel("X")
            plt.ylabel("Y")

        # Find the first and last points of drop mapped to original
        first_change_idx = mapped_change_indices[0]
        last_change_idx = mapped_change_indices[-1]

        # Exclude data from first to last point of drop
        x_concatenated = np.concatenate((x[:first_change_idx], x[last_change_idx + 1:]))
        y_concatenated = np.concatenate((y[:first_change_idx], y[last_change_idx + 1:]))

        # Create an envelope function for the concatenated data
        upper_envelope, envelope = self.create_envelope(x_concatenated, y_concatenated)

        # Fit a polynomial to the envelope
        poly_degree = 4
        poly_coeffs = np.polyfit(x_concatenated, envelope, poly_degree)

        if verbose:
            plt.figure()
            plt.plot(x, y, linewidth=0.1, label='Calibration reference')
            poly_fit = np.polyval(poly_coeffs, x)
            plt.plot(x, poly_fit, 'r', label='Polynomial Fit (Envelope)')
            plt.plot(x[first_change_idx:last_change_idx + 1], y[first_change_idx:last_change_idx + 1], 'k--',
                     label='Excluded Region')
            plt.legend(loc=0)
            plt.title("Final generated polynomial for fitting")
            plt.xlabel("X")
            plt.ylabel("Y")

        # Calibrate the data by subtracting the polynomial fit (envelope)
        calibrated_data = self.calibrate_data(x, y, poly_coeffs)

        if verbose:
            plt.figure()
            plt.plot(x, y, linewidth=1, label='Calibrated input response')
            plt.plot(x, calibrated_data, 'g', label='Calibrated envelope response')
            plt.legend(loc=0)
            plt.title("Final calibration")
            plt.xlabel("X")
            plt.ylabel("Y")

        poly_fit = np.polyval(poly_coeffs, x)
        x_envelope = x
        y_envelope = poly_fit

        calibrated = calibrated_data

        return calibrated, x_envelope, y_envelope
