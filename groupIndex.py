# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 01:37:33 2023

@author: musta
"""

import sys
sys.path.append(r'D:\Academics\PyCharmProjects')  # Add the directory to sys.path
import siepic_analysis_package as siap
import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

font = {'family': 'normal',
        'size': 18}

# Variables to adjust
directory_path = "D:\Academics\PyCharmProjects\Data\Actives-May-2023"
wavl = 1550
fname_data = directory_path + "\\" + str(wavl) + "_TE"  # filename containing the desired data
device_prefix = "PCM_DC_Length"
device_suffix = "um_2"
port_cross = 1  # port containing the cross-port data to process
port_bar = 0  # port containing the bar-port data to process

DL = 53.793e-6  # delta-length in the MZI used for these test structures
wavl_range = [1460, 1580]  # adjust wavelength range
window = 210  # filtering window, cleaner data is easier to detect peaks
peak_prominence = 0.25  # adjust of peak detection is bad




class GroupIndex:
    def __init__(self, fname_data, device_prefix, device_suffix, window, peak_prominence, DL, port_cross, wavl_range):
        self.fname_data = fname_data
        self.device_prefix = device_prefix
        self.device_suffix = device_suffix
        self.window = window
        self.peak_prominence = peak_prominence
        self.DL = DL
        self.port_cross = port_cross
        self.wavl_range = wavl_range
        self.devices = []

    def getDeviceParameter(self, deviceID, devicePrefix, deviceSuffix=''):
        parameter = float(deviceID.removeprefix(devicePrefix).removesuffix(deviceSuffix).replace('p', '.'))
        return parameter

    def extract_periods(self, wavelength, transmission, min_prominence=.25, plot=False):
        transmission_centered = transmission - np.mean(transmission)
        peak_indices = find_peaks(transmission_centered, prominence=min_prominence)[0]
        peak_wavelengths = wavelength[peak_indices]
        periods = np.diff(peak_wavelengths)

        inverted_transmission_centered = -transmission_centered
        trough_indices = find_peaks(inverted_transmission_centered, prominence=min_prominence)[0]
        trough_wavelengths = wavelength[trough_indices]

        extinction_ratios = []
        for i in range(len(peak_indices) - 1):
            trough_value = transmission[trough_indices[i]]
            peak_value = transmission[peak_indices[i]]
            extinction_ratios.append(np.abs(peak_value - trough_value))

        midpoints = (peak_wavelengths[:-1] + peak_wavelengths[1:]) / 2
        periods_at_midpoints = dict(zip(midpoints, periods))
        extinction_ratios_at_midpoints = dict(zip(midpoints, extinction_ratios))

        if plot:
            fig, axs = plt.subplots(3, figsize=(14, 20))
            axs[0].plot(wavelength, transmission, label="Signal")
            axs[0].plot(peak_wavelengths, transmission[peak_indices], "x", label="Peaks")
            axs[0].plot(trough_wavelengths, transmission[trough_indices], "x", label="Troughs")
            axs[0].set_title("Signal with Detected Peaks")
            axs[0].legend()

            axs[1].scatter(midpoints, periods)
            axs[1].set_xlabel("Wavelength")
            axs[1].set_ylabel("FSR")
            axs[1].set_title("FSR as a function of Wavelength")

            axs[2].scatter(midpoints, extinction_ratios)
            axs[2].set_xlabel("Wavelength")
            axs[2].set_ylabel("Extinction Ratio (dB)")
            axs[2].set_title("Extinction Ratio as a function of Wavelength")

            plt.tight_layout()
            plt.show()

        return midpoints, periods, extinction_ratios

    def average_arrays(self, x_values, y_values_list, x_new, plot=False):
        y_values_interp_list = []

        for x, y in zip(x_values, y_values_list):
            f = interp1d(x, y, bounds_error=False, fill_value=np.nan)
            y_new = f(x_new)
            y_values_interp_list.append(y_new)

        y_values_interp_array = np.array(y_values_interp_list)
        y_average = np.nanmean(y_values_interp_array, axis=0)

        mask = np.isnan(y_average)
        y_average[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_average[~mask])

        y_std = np.nanstd(y_values_interp_array, axis=0)
        mask_std = np.isnan(y_std)
        y_std[mask_std] = np.interp(np.flatnonzero(mask_std), np.flatnonzero(~mask_std), y_std[~mask_std])

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(x_new, y_average, 'k-', label='Average')
            plt.fill_between(x_new, y_average - y_std, y_average + y_std, color='gray', alpha=0.2, label='Std dev')
            plt.title('Averaged Data')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend(loc='best')
            plt.grid(True)
            plt.show()

        return x_new, y_average, y_std, y_average

    def process_device_data(self):
        for root, dirs, files in os.walk(self.fname_data):
            if os.path.basename(root).startswith(self.device_prefix):
                for file in files:
                    if file.endswith(".csv"):
                        device = siap.analysis.processCSV(os.path.join(root, file))
                        self.devices.append(device)

                        device.length = self.getDeviceParameter(device.deviceID, self.device_prefix, self.device_suffix)

                        device.wavl, device.pwr[self.port_cross] = siap.analysis.truncate_data(device.wavl,
                                                                                               siap.core.smooth(
                                                                                                   device.wavl,
                                                                                                   device.pwr[
                                                                                                       self.port_cross],
                                                                                                   window=self.window),
                                                                                               self.wavl_range[0],
                                                                                               self.wavl_range[1])
                        [device.cross_T, device.fit] = siap.analysis.baseline_correction(
                            [device.wavl, device.pwr[self.port_cross]])

                        midpoints, fsr, extinction_ratios = self.extract_periods(device.wavl, device.cross_T,
                                                                                 min_prominence=self.peak_prominence,
                                                                                 plot=False)

                        device.ng_wavl = midpoints
                        device.ng = siap.analysis.getGroupIndex([i * 1e-9 for i in device.ng_wavl],
                                                                [i * 1e-9 for i in fsr],
                                                                delta_length=self.DL)

                        device.kappa = []
                        for er in extinction_ratios:
                            device.kappa.append(0.5 - 0.5 * np.sqrt(1 / 10 ** (er / 10)))
        return self.devices

    def plot_group_index(self):
        # Simulated data coefficients
        ng_500nm_fit = [-4.58401408e-07, 7.63213215e-05, -1.90478033e-03, 4.11962711e+00]
        ng_wavl_fit = [7.71116919e-16, 8.52254170e-13, 1.12019032e-09, 1.45999992e-06]

        # Simulated data
        ng_500nm = np.polyval(ng_500nm_fit, np.linspace(0, 100 - 1, 100))
        wavl_sim = np.polyval(ng_wavl_fit, np.linspace(0, 100 - 1, 100)) * 1e9

        fig, ax1 = plt.subplots(figsize=(10, 6))

        for device in self.devices:
            ax1.scatter(device.ng_wavl, device.ng, color='black', linewidth=0.1)

        ax1.plot(wavl_sim, ng_500nm, color='blue', label='Simulated 500 nm X 220 nm')

        ax1.set_xlim(np.min([np.min(i.ng_wavl) for i in self.devices]),
                     np.max([np.max(i.ng_wavl) for i in self.devices]))

        ng_avg_wavl, ng_avg, ng_std, ng_std_avg = self.average_arrays([i.ng_wavl for i in self.devices],
                                                                      [i.ng for i in self.devices],
                                                                      np.linspace(self.wavl_range[0],
                                                                                  self.wavl_range[1]))
        ax1.plot(ng_avg_wavl, ng_avg, '--', color='black', label='Average')
        ax1.fill_between(np.linspace(self.wavl_range[0], self.wavl_range[1]), ng_std_avg - ng_std, ng_std_avg + ng_std,
                         color='gray',
                         alpha=0.2, label='Std dev')
        ax1.legend()
        ax1.set_ylabel('Group index')
        ax1.set_xlabel('Wavelength [nm]')

        plt.show()

    def sort_devices_by_length(self):
        sorted_devices = sorted(self.devices, key=lambda d: d.length)
        return sorted_devices

    def plot_coupling_coefficient_contour(self):
        ng_wavl = np.array([device.ng_wavl[:10] for device in self.devices])
        device_lengths = np.array([device.length for device in self.devices])

        common_ng_wavl = np.unique(np.concatenate(ng_wavl))
        X, Y = np.meshgrid(common_ng_wavl, device_lengths)
        Z = np.empty_like(X)

        for i, device in enumerate(self.devices):
            interp_func = interp1d(device.ng_wavl, device.kappa, kind='linear', fill_value='extrapolate')
            Z[i, :] = interp_func(common_ng_wavl)

        fig, ax = plt.subplots(figsize=(10, 6))
        contour = ax.contourf(X, Y, Z, cmap='viridis')

        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Coupling Length [µm]")
        ax.set_title("Coupling Coefficient Contour Map (100 nm gap)")

        fig.colorbar(contour)
        plt.show()

# Sample usage
parameters = {
    'fname_data': fname_data,  # Replace with the actual path to your data
    'device_prefix': device_prefix,
    'device_suffix': device_suffix,
    'window': window,
    'peak_prominence': peak_prominence,
    'DL': DL,
    'port_cross': port_cross,
    'wavl_range': wavl_range
}

group_index_instance = GroupIndex(**parameters)
group_index_instance.process_device_data()
group_index_instance.plot_group_index()
group_index_instance.plot_coupling_coefficient_contour()
