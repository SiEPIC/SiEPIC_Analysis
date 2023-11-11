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
sys.path.append(r'D:\Academics\PyCharmProjects')  # Add the directory to sys.path
import siepic_analysis_package as siap
import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np

# Variables to adjust
directory_path = "D:\Academics\PyCharmProjects\Data\Actives-May-2023"
wavl = 1550
fname_data = directory_path + "\\" + str(wavl) + "_TE"  # filename containing the desired data
device_prefix = "PCM_Bragg_C__1000N"
device_suffix = "nmPeriod500nmW20nmdW0Apo"
port_drop = 1 # port in the measurement set containing the drop port data
port_thru = 1 # port in the measurement set containing the through port data

class DirectionalCoupler:
    def __init__(self, fname_data, device_prefix, port_thru, port_drop, device_suffix, tol=3, N_seg=325):
        self.fname_data = fname_data
        self.device_prefix = device_prefix
        self.port_thru = port_thru
        self.port_drop = port_drop
        self.device_suffix = device_suffix
        self.tol = tol
        self.N_seg = N_seg

        self.devices = []
        self.period = []
        self.WL = []
        self.BW = []

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
        plt.figure()
        for device in self.devices:
            label = 'Period = ' + str(self.getDeviceParameter(device.deviceID, self.device_prefix, self.device_suffix)) + ' nm'
            plt.plot(device.wavl, device.pwr[self.port_drop], label=label)

        plt.legend(loc=0)
        plt.ylabel('Power [dBm]', color='black')
        plt.xlabel('Wavelength [nm]', color='black')
        plt.title("Raw measurement of all structures")
        plt.savefig('devices_raw' + '.pdf')
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        plt.figure()
        for device in self.devices:
            label = 'Period = ' + str(self.getDeviceParameter(device.deviceID, self.device_prefix, self.device_suffix)) + ' nm'
            plt.plot(device.wavl, device.dropCalib, label=label)

        plt.legend(loc=0)
        plt.ylabel('Transmission (dB)', color='black')
        plt.xlabel('Wavelength (nm)', color='black')
        plt.title("Calibrated measurement of all structures (using envelope calibration)")
        plt.savefig('devices_calib' + '.pdf')
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

    def plot_analysis_results(self):
        fig, ax1 = plt.subplots()

        ax1.scatter(self.period, self.WL, color='blue')
        ax1.set_xlabel('Grating period [nm]')
        ax1.set_ylabel('Bragg wavelength [nm]', color='blue')
        ax1.tick_params(axis='y', colors='blue')

        ax2 = ax1.twinx()
        ax2.scatter(self.period, self.BW, color='red')
        ax2.set_ylabel('3 dB Bandwidth [nm]', color='red')
        ax2.tick_params(axis='y', colors='red')

        plt.title("Extracted bandwidth and central wavelength of the Bragg gratings")
        plt.savefig('analysis' + '.pdf')
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})
        plt.show()

    def overlay_simulation_data(self, sim_label = 'Simulation (SiO2 Clad)'):
        simulation_period_sio2 = [313, 315, 317, 319, 321, 323]
        simulation_wavl_sio2 = [1536.64, 1542.24, 1547.85, 1553.45, 1559.06, 1564.56]

        plt.figure()
        plt.scatter(self.period, self.WL, color='r', marker='x', label='Experiment')
        plt.scatter(simulation_period_sio2, simulation_wavl_sio2, color='b', marker='o', label=sim_label)
        plt.legend()
        plt.ylabel('Bragg Wavelength [nm]', color='black')
        plt.xlabel('Grating Period [nm]', color='black')
        plt.title("Comparison of Bragg wavelength between simulation and experiment.")
        plt.savefig('analysis_WL' + '.pdf')
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})
        plt.show()


# Example usage:
dc = DirectionalCoupler(
    fname_data= fname_data,
    device_prefix= device_prefix,
    port_thru= port_thru,
    port_drop= port_drop,
    device_suffix= device_suffix
)

dc.process_files()
dc.plot_devices()
dc.plot_analysis_results()
dc.overlay_simulation_data()