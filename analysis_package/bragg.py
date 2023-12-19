"""
SiEPIC Analysis Package

Author:     Mustafa Hammood
            Mustafa@siepic.com

Example:    Application of SiEPIC_AP analysis functions
            Process data of various contra-directional couplers (CDCs)
            Extract the period and bandwidth from a set of devices
"""
#%%
import sys
import io
sys.path.append(r'D:\Academics\PyCharmProjects')  # Add the directory to sys.path
import siepic_analysis_package as siap
import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np
import pandas as pd

class DirectionalCoupler:
    def __init__(self, fname_data, device_prefix, port_thru, port_drop, device_suffix,
                 name, wavl, pol, main_script_directory,
                 tol=3, N_seg=325):
        self.fname_data = fname_data
        self.device_prefix = device_prefix
        self.port_thru = port_thru
        self.port_drop = port_drop
        self.device_suffix = device_suffix
        self.name = name
        self.wavl = wavl
        self.pol = pol
        self.main_script_directory = main_script_directory

        self.tol = tol
        self.N_seg = N_seg

        self.devices = []
        self.period = []
        self.WL = []
        self.BW = []
        self.df_figures = pd.DataFrame()


    def getDeviceParameter(self, deviceID, devicePrefix, deviceSuffix=''):
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
        parameter = float(deviceID.removeprefix(devicePrefix).removesuffix(deviceSuffix))
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
                        self.period.append(self.getDeviceParameter(device.deviceID, self.device_prefix, self.device_suffix))
                        self.WL.append(device.WL)
                        self.BW.append(device.BW)

        return self.devices, self.period, self.WL, self.BW

    def plot_devices(self):
        plt.figure(figsize=(10, 6))
        for device in self.devices:
            label = 'Period = ' + str(self.getDeviceParameter(device.deviceID, self.device_prefix, self.device_suffix)) + ' nm'
            plt.plot(device.wavl, device.pwr[self.port_drop], label=label)

        plt.legend(loc=0)
        plt.ylabel('Power [dBm]', color='black')
        plt.xlabel('Wavelength [nm]', color='black')
        plt.title("Raw measurement of all structures")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        # save plots
        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_devices_raw, format='pdf')
        # plt.show()  # Display graph

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Directly append the figure information to the existing DataFrame
        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_raw', 'Figure': img_buffer},
            ignore_index=True
        )

        plt.figure(figsize=(10, 6))
        for device in self.devices:
            label = 'Period = ' + str(self.getDeviceParameter(device.deviceID, self.device_prefix, self.device_suffix)) + ' nm'
            plt.plot(device.wavl, device.dropCalib, label=label)

        plt.legend(loc=0)
        plt.ylabel('Transmission (dB)', color='black')
        plt.xlabel('Wavelength (nm)', color='black')
        plt.title("Calibrated measurement of all structures (using envelope calibration)")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        # save plots
        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_devices_calib, format='pdf')
        # plt.show()  # Display graph

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Directly append the figure information to the existing DataFrame
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

        # save plots
        pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL = self.saveGraph()
        plt.savefig(pdf_path_analysis, format='pdf')
        # plt.show()  # Display graph

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Directly append the figure information to the existing DataFrame
        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_central', 'Figure': img_buffer},
            ignore_index=True
        )

    def overlay_simulation_data(self, target_wavelength, sim_label = 'Simulation (SiO2 Clad)'):
        simulation_period_sio2 = [313, 315, 317, 319, 321, 323]
        simulation_wavl_sio2 = [1536.64, 1542.24, 1547.85, 1553.45, 1559.06, 1564.56]

        # Interpolate simulation period at target_wavelength_sim
        simulation_period_at_target_sim = np.interp(target_wavelength, simulation_wavl_sio2, simulation_period_sio2)

        # Interpolate experimental period at target_wavelength_exp
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
        common_wavelengths = np.linspace(min(min(self.period), min(simulation_period_sio2)),
                                         max(max(self.period), max(simulation_period_sio2)), 100)

        # Interpolate both experimental and simulated data to the common set of wavelengths
        exp_wavelength_interp = np.interp(common_wavelengths, self.period, self.WL)
        sim_wavelength_interp = np.interp(common_wavelengths, simulation_period_sio2, simulation_wavl_sio2)

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

        # Find the average difference
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

        # Directly append the figure information to the existing DataFrame
        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_overlay', 'Figure': img_buffer},
            ignore_index=True
        )

        # Return the average difference and the combined DataFrame
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

        # Combine the directory and the filename to get the full paths
        pdf_path_devices_raw = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_devices_raw.pdf")
        pdf_path_devices_calib = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_devices_calib.pdf")
        pdf_path_analysis = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_analysis.pdf")
        pdf_path_analysis_WL = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_analysis_WL.pdf")

        # Now, you can save your PDFs to pdf_path_raw and pdf_path_cutback
        return pdf_path_devices_raw, pdf_path_devices_calib, pdf_path_analysis, pdf_path_analysis_WL
