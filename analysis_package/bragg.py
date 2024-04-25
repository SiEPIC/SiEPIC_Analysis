
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
#%%
import os
import sys
import io

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import siepic_analysis_package as siap

import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np
import pandas as pd

class DirectionalCoupler:
    def __init__(self, fname_data, device_prefix, port_thru, port_drop, device_suffix,
                 name, wavl, pol, main_script_directory,
                 tol, N_seg):
        self.fname_data = fname_data
        self.device_prefix = device_prefix
        self.port_thru = port_thru
        self.port_drop = port_drop
        self.device_suffix = device_suffix
        self.name = name
        self.wavl = wavl
        self.pol = pol
        self.main_script_directory = main_script_directory

        if tol is None:
            self.tol = 4
        else: self.tol = tol
        if N_seg is None:
            self.N_seg = 325
        else: self.N_seg = N_seg

        self.devices = []
        self.period = []
        self.WL = []
        self.BW = []
        self.df_figures = pd.DataFrame()


    def getDeviceParameter(self, deviceID):
        """Find the variable parameter of a device based on the ID

        IMPORTANT: "removeprefix" and "removesuffix" are only available
            for Python >= 3.9

        Args:
            deviceID (string): ID of the device.
            devicePrefix (string): Prefix string in the device that's before the variable parameter
            deviceSuffix (string): Any additional fields in the suffix of a device that need to be stripped, optional.

        Returns:
            parameter (float): variable parameter of the device (unit based on whats in the ID)
        """
        parameter = float(deviceID.removeprefix(self.device_prefix).removesuffix(self.device_suffix))
        return parameter

    def process_files(self):
        for root, dirs, files in os.walk(self.fname_data):
            if os.path.basename(root).startswith(self.device_prefix):
                for file in files:
                    if file.endswith(".csv"):
                        file_path = os.path.join(root, file)
                        device = siap.analysis.processCSV(file_path)

                        device.dropCalib, device.ThruEnvelope, x, y = siap.analysis.calibrate_envelope(
                            device.wavl, device.pwr[self.port_thru], device.pwr[self.port_drop],
                            N_seg=self.N_seg, tol=self.tol, verbose=False)

                        [device.BW, device.WL] = siap.analysis.bandwidth(device.wavl, -device.dropCalib, threshold=6)

                        self.devices.append(device)
                        self.period.append(self.getDeviceParameter(device.deviceID))
                        self.WL.append(device.WL)
                        self.BW.append(device.BW)

        return self.devices, self.period, self.WL, self.BW

    def plot_devices(self):
        plt.figure(figsize=(10, 6))
        for device in self.devices:
            label = 'Period = ' + str(self.getDeviceParameter(device.deviceID)) + ' nm'
            plt.plot(device.wavl, device.pwr[self.port_drop], label=label)

        plt.legend(loc=0)
        plt.ylabel('Power [dBm]', color='black')
        plt.xlabel('Wavelength [nm]', color='black')
        plt.title("Raw measurement of all structures")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_devices_raw, format='pdf')
        # plt.show()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_raw', 'Figure': img_buffer},
            ignore_index=True
        )

        plt.figure(figsize=(10, 6))
        for device in self.devices:
            label = 'Period = ' + str(self.getDeviceParameter(device.deviceID)) + ' nm'
            plt.plot(device.wavl, device.dropCalib, label=label)

        plt.legend(loc=0)
        plt.ylabel('Transmission (dB)', color='black')
        plt.xlabel('Wavelength (nm)', color='black')
        plt.title("Calibrated measurement of all structures (using envelope calibration)")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_devices_calib, format='pdf')
        # plt.show()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_calib', 'Figure': img_buffer},
            ignore_index=True
        )

    def plot_analysis_results(self):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.scatter(self.period, self.WL, color='blue')
        ax1.set_xlabel('Grating period [nm]')
        ax1.set_ylabel('Bragg wavelength [nm]', color='blue')
        ax1.tick_params(axis='y', colors='blue')

        ax2 = ax1.twinx()
        ax2.scatter(self.period, self.BW, color='red')
        ax2.set_ylabel('3 dB Bandwidth [nm]', color='red')
        ax2.tick_params(axis='y', colors='red')

        plt.title("Extracted bandwidth and central wavelength of the Bragg gratings")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_analysis, format='pdf')
        # plt.show()

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
        simulation_period_sio2_1310 = [273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283]
        simulation_wavl_sio2_1310 = [1318.45, 1321.05, 1323.65, 1326.26, 1328.86, 1331.46, 1334.06, 1336.54, 1339.14,
                                     1341.7, 1344.21]

        # 1310nm simulation results (220 nm SOI, air)
        period_sim_air_1310 = [273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283]
        wavl_sim_air_1310 = [1295.48, 1297.79, 1300.2, 1302.52, 1304.94, 1307.26, 1309.57, 1311.88, 1314.08, 1316.4,
                             1318.82]

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

        simulation_period_at_target_sim = np.interp(target_wavelength, simulation_wavl, simulation_period)
        experimental_period_at_target_exp = np.interp(target_wavelength, self.WL, self.period)

        print(f"Simulation period at {target_wavelength} nm: {simulation_period_at_target_sim} nm")
        print(f"Experimental period at {target_wavelength} nm: {experimental_period_at_target_exp} nm")

        plt.figure(figsize=(10, 6))
        plt.scatter(self.period, self.WL, color='r', marker='x', label='Experiment')
        plt.scatter(simulation_period_sio2, simulation_wavl_sio2, color='b', marker='o', label=sim_label)

        plt.legend()
        plt.ylabel('Bragg Wavelength [nm]', color='black')
        plt.xlabel('Grating Period [nm]', color='black')
        plt.title("Comparison of Bragg wavelength between simulation and experiment.")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

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

        exp_wavelength_fit = exp_poly_func(common_wavelengths)
        sim_wavelength_fit = sim_poly_func(common_wavelengths)
        differences = exp_wavelength_fit - sim_wavelength_fit
        average_difference = np.mean(differences)

        print(f"Bragg Grating Wavelength Drift is: {average_difference} nm for {self.name}_{self.pol}{self.wavl}")

        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_analysis_WL, format='pdf')
        # plt.show()

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
        output_directory = os.path.join(self.main_script_directory, f"{self.name}_{self.pol}{self.wavl}")
        os.makedirs(output_directory, exist_ok=True)

        pdf_path_devices_raw = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_devices_raw.pdf")
        pdf_path_devices_calib = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_devices_calib.pdf")
        pdf_path_analysis = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_analysis.pdf")
        pdf_path_analysis_WL = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_analysis_WL.pdf")

        return pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL
