# -*- coding: utf-8 -*-
import sys
sys.path.append(r'D:\Academics\PyCharmProjects')  # Add the directory to sys.path
import siepic_analysis_package as siap
import matplotlib.pyplot as plt
import matplotlib, os
import numpy as np
import io
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

class GroupIndex:
    def __init__(self, directory_path, wavl, pol, device_prefix, device_suffix, port_cross, port_bar,
                 name, main_script_directory,
                 DL=53.793e-6, wavl_range=None, window=210, peak_prominence=0.25):
        if wavl_range is None:
            wavl_range = [1460, 1580]
        self.directory_path = directory_path
        self.wavl = wavl
        self.pol = pol
        self.device_prefix = device_prefix
        self.device_suffix = device_suffix
        self.port_cross = port_cross
        self.port_bar = port_bar
        self.DL = DL
        self.wavl_range = wavl_range
        self.window = window
        self.peak_prominence = peak_prominence
        self.devices = []
        self.name = name
        self.main_script_directory = main_script_directory
        self.df_figures = pd.DataFrame()

    def _get_device_parameter(self, deviceID):
        try:
            start_index = deviceID.index(self.device_prefix) + len(self.device_prefix)
            end_index = deviceID.index(self.device_suffix, start_index)
            parameter = float(deviceID[start_index:end_index])
            return parameter

        except ValueError:
            # Handle the case where prefix or suffix is not found
            return None  # Return None to indicate failure

    def _extract_periods(self, wavelength, transmission, min_prominence=.25, plot=False):
        # Subtract the mean of the signal
        transmission_centered = transmission - np.mean(transmission)

        # Find peaks
        peak_indices = find_peaks(transmission_centered, prominence=min_prominence)[0]
        peak_wavelengths = wavelength[peak_indices]

        # Calculate periods
        periods = np.diff(peak_wavelengths)

        # Find troughs
        inverted_transmission_centered = -transmission_centered
        trough_indices = find_peaks(inverted_transmission_centered, prominence=min_prominence)[0]
        trough_wavelengths = wavelength[trough_indices]

        # Calculate extinction ratios
        extinction_ratios = []
        for i in range(len(peak_indices) - 1):
            # find troughs between current peak and next peak
            trough_value = transmission[trough_indices[i]]
            peak_value = transmission[peak_indices[i]]
            extinction_ratios.append(np.abs(peak_value - trough_value))

        # Record the period and extinction ratio at the midpoint between each pair of consecutive peaks
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
            # plt.show()

        return midpoints, periods, extinction_ratios

    def _average_arrays(self, x_values, y_values_list, x_new, plot=False):
        """
        x_values: list of arrays, each containing the x-values for one of the y-value arrays
        y_values_list: list of arrays, the y-value arrays to be averaged
        x_new: array, the new common x-grid to interpolate onto
        plot: boolean, if True the function will create a plot of the averaged data with error bars
        """
        from scipy.interpolate import interp1d
        import numpy as np

        # List to store the interpolated y-values
        y_values_interp_list = []

        # Interpolate each y-value array onto the new x-grid
        for x, y in zip(x_values, y_values_list):
            f = interp1d(x, y, bounds_error=False, fill_value=np.nan)
            y_new = f(x_new)
            y_values_interp_list.append(y_new)

        # Convert the list of interpolated y-value arrays into a 2D array
        y_values_interp_array = np.array(y_values_interp_list)

        # Compute the mean of the interpolated y-value arrays, ignoring NaNs
        y_average = np.nanmean(y_values_interp_array, axis=0)

        # Replace any remaining NaNs in the average with the closest valid value
        mask = np.isnan(y_average)
        y_average[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_average[~mask])

        # Compute the standard deviation of the interpolated y-value arrays, ignoring NaNs
        y_std = np.nanstd(y_values_interp_array, axis=0)

        # Replace any NaNs in the standard deviation with the closest valid value
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
            # plt.show()

        return x_new, y_average, y_std, y_average

    def process_device_data(self):
        for root, dirs, files in os.walk(self.directory_path):
            if os.path.basename(root).startswith(self.device_prefix):
                for file in files:
                    if file.endswith(".csv"):
                        device = siap.analysis.processCSV(os.path.join(root, file))
                        self.devices.append(device)

                        device.length = self._get_device_parameter(device.deviceID)
                        device.wavl, device.pwr[self.port_cross] = siap.analysis.truncate_data(device.wavl,
                                                                                            siap.core.smooth(device.wavl,
                                                                                                             device.pwr[self.port_cross],
                                                                                                             window=self.window),
                                                                                            self.wavl_range[0], self.wavl_range[1])
                        [device.cross_T, device.fit] = siap.analysis.baseline_correction(
                                [device.wavl, device.pwr[self.port_cross]])
                        midpoints, fsr, extinction_ratios = self._extract_periods(device.wavl, device.cross_T,
                                                                                min_prominence=self.peak_prominence, plot=False)

                        device.ng_wavl = midpoints
                        device.ng = siap.analysis.getGroupIndex([i * 1e-9 for i in device.ng_wavl],
                                                                [i * 1e-9 for i in fsr],
                                                                delta_length=self.DL)

                        device.kappa = []
                        for er in extinction_ratios:
                            device.kappa.append(0.5 - 0.5 * np.sqrt(1 / 10 ** (er / 10)))

    def plot_group_index(self, target_wavelength):
        """
        Plot group index for multiple devices along with simulated data.

        Returns:
        - None
        """
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

        ng_avg_wavl, ng_avg, ng_std, ng_std_avg = self._average_arrays([i.ng_wavl for i in self.devices],
                                                                       [i.ng for i in self.devices],
                                                                       np.linspace(np.min(wavl_sim), np.max(wavl_sim)))

        # Convert target_wavelength to the same type as ng_avg_wavl
        target_wavelength = np.asarray(target_wavelength, dtype=ng_avg_wavl.dtype)

        # Find the index in ng_avg_wavl closest to target_wavelength
        idx = np.argmin(np.abs(ng_avg_wavl - target_wavelength))

        # Print ng_avg and error bar at target_wavelength
        print(f"Group Index at {target_wavelength} nm is {ng_avg[idx]} ± {ng_std[idx]} for {self.name}_{self.pol}{self.wavl}")

        gindex = ng_avg[idx]
        gindexError = ng_std[idx]

        ax1.fill_between(np.linspace(np.min(wavl_sim), np.max(wavl_sim)), ng_std_avg - ng_std, ng_std_avg + ng_std,
                         color='gray', alpha=0.2, label='Std dev')
        ax1.legend()
        ax1.set_ylabel('Group index')
        ax1.set_xlabel('Wavelength [nm]')

        pdf_path_gindex, pdf_path_contour = self.saveGraph()
        plt.savefig(pdf_path_gindex, format='pdf')
        # plt.show()  # Display the plot

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Directly append the figure information to the existing DataFrame
        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_gsim', 'Figure': img_buffer},
            ignore_index=True
        )

        return gindex, gindexError

    def sort_devices_by_length(self):
        self.devices = sorted(self.devices, key=lambda d: d.length)

    from scipy.interpolate import interp1d

    def plot_coupling_coefficient_contour(self):
        """
        Plot a contour map of coupling coefficients.

        Returns:
        - None
        """
        # Extracting wavelength and device length data
        ng_wavl = [device.ng_wavl[:11] for device in self.devices]
        device_lengths = [device.length for device in self.devices]

        # Creating a common wavelength grid
        common_ng_wavl = np.unique(np.concatenate(ng_wavl))

        # Sort the common_ng_wavl array
        common_ng_wavl.sort()

        # Creating meshgrid for wavelength and device length
        X, Y = np.meshgrid(common_ng_wavl, device_lengths)

        # Creating an empty 2D array for coupling coefficient data
        Z = np.empty((len(self.devices), len(common_ng_wavl)))

        # Populating the 2D array with coupling coefficient data
        for i, device_ng_wavl in enumerate(ng_wavl):
            # Take a specific wavelength from device_ng_wavl
            wavelength_at_i = device_ng_wavl[0]

            # Sort ng_wavl and kappa arrays before interpolation
            sorted_indices = np.argsort(device_ng_wavl)
            interp_func = interp1d(np.array(device_ng_wavl)[sorted_indices],
                                   np.array(self.devices[i].kappa)[sorted_indices], kind='linear',
                                   fill_value='extrapolate')
            Z[i, :] = interp_func(common_ng_wavl)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting the contour map
        contour = ax.contourf(X, Y, Z, cmap='viridis')

        # Adding labels and title to the plot
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Coupling Length [µm]")
        ax.set_title("Coupling Coefficient Contour Map (100 nm gap)")

        # Adding a colorbar
        fig.colorbar(contour)

        # Displaying the plot
        pdf_path_gindex, pdf_path_contour = self.saveGraph()
        plt.savefig(pdf_path_contour, format='pdf')

        # plt.show()  # Display the plot

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Directly append the figure information to the existing DataFrame
        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_coup', 'Figure': img_buffer},
            ignore_index=True
        )

    def saveGraph(self):
        """
        Save a graph as PDF files in a directory based on the `self.name` and `self.pol` attributes.

        This method creates a directory named after `self.name` and `self.pol` inside the specified `main_script_directory`
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
        pdf_path_gindex = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_gindex.pdf")
        pdf_path_contour = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_contour.pdf")

        # Now, you can save your PDFs
        return pdf_path_gindex, pdf_path_contour
