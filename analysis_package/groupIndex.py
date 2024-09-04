import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import siepic_analysis_package as siap

import matplotlib.pyplot as plt
import os
import numpy as np
import io
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

class GroupIndex:
    def __init__(self, directory_path, wavl, pol, device_prefix, device_suffix, port_cross, port_bar,
                 name, main_script_directory, measurement_label, wavl_range, DL,
                 peak_prominence, window=210):

        if peak_prominence is None:
            self.peak_prominence = 0.25
        else:
            self.peak_prominence = peak_prominence

        self.directory_path = directory_path
        self.wavl = wavl
        self.pol = pol
        self.device_prefix = device_prefix
        self.device_suffix = device_suffix
        self.port_cross = port_cross
        self.port_bar = port_bar
        self.window = window
        self.label = measurement_label
        self.wavl_range = wavl_range
        self.DL = DL

        self.devices = []
        self.name = name
        self.main_script_directory = main_script_directory
        self.df_figures = pd.DataFrame()

    def _get_device_parameter(self, deviceID):
        parameter = float(deviceID.removeprefix(self.device_prefix).removesuffix(self.device_suffix).replace('p', '.'))
        return parameter

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

    def average_arrays(self, x_values, y_values_list, x_new, plot=False):
        """
        x_values: list of arrays, each containing the x-values for one of the y-value arrays
        y_values_list: list of arrays, the y-value arrays to be averaged
        x_new: array, the new common x-grid to interpolate onto
        plot: boolean, if True the function will create a plot of the averaged data with error bars
        """
        from scipy.interpolate import interp1d
        import numpy as np

        y_values_interp_list = []

        # Interpolate each y-value array onto the new x-grid
        for x, y in zip(x_values, y_values_list):
            if len(x) == 0 or np.all(np.isnan(y)):
                y_new = np.full_like(x_new, np.nan)  # Handle case where y is all NaNs
            else:
                f = interp1d(x, y, bounds_error=False, fill_value=np.nan)
                y_new = f(x_new)
            y_values_interp_list.append(y_new)

        # Convert the list of interpolated y-value arrays into a 2D array
        y_values_interp_array = np.array(y_values_interp_list)

        # Compute the mean of the interpolated y-value arrays, ignoring NaNs
        y_average = np.nanmean(y_values_interp_array, axis=0)
        mask = np.isnan(y_average)

        # Only perform interpolation if there are non-NaN values
        if np.any(~mask):
            y_average[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_average[~mask])
        else:
            print("Warning: No non-NaN data available for interpolation.")
            return x_new, np.full_like(x_new, np.nan), np.full_like(x_new, np.nan), np.full_like(x_new, np.nan)

        y_std = np.nanstd(y_values_interp_array, axis=0)

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

    def process_device_data(self, min, max):
        for root, dirs, files in os.walk(self.directory_path):
            if os.path.basename(root).startswith(self.device_prefix):
                for file in files:
                    if file.endswith(".csv"):
                        device = siap.analysis.processCSV(os.path.join(root, file))
                        self.devices.append(device)

                        device.length = self._get_device_parameter(device.deviceID)

                        if min:
                            range_min = min
                        else:
                            range_min = self.wavl_range[0]
                        if max:
                            range_max = max
                        else:
                            range_max = self.wavl_range[1]

                        # Truncate the device data
                        device.wavl, device.pwr[self.port_cross] = siap.analysis.truncate_data(
                            device.wavl,
                            siap.core.smooth(device.wavl, device.pwr[self.port_cross], window=self.window),
                            range_min,
                            range_max
                        )
                        # Ensure that truncated data is not empty
                        if len(device.wavl) == 0 or len(device.pwr[self.port_cross]) == 0:
                            print(f"Warning: No data left after truncation for device {device.deviceID}")
                            continue

                        [device.cross_T, device.fit] = siap.analysis.baseline_correction(
                            [device.wavl, device.pwr[self.port_cross]]
                        )
                        midpoints, fsr, extinction_ratios = self._extract_periods(
                            device.wavl,
                            device.cross_T,
                            min_prominence=self.peak_prominence,
                            plot=False
                        )

                        # plt.show()

                        device.ng_wavl = midpoints
                        device.ng = siap.analysis.getGroupIndex(
                            [i * 1e-9 for i in device.ng_wavl],  # Convert wavelengths to meters
                            [i * 1e-9 for i in fsr],  # Convert FSR to meters
                            delta_length=self.DL
                        )

                        device.kappa = []
                        for er in extinction_ratios:
                            device.kappa.append(0.5 - 0.5 * np.sqrt(1 / 10 ** (er / 10)))

    def plot_group_index(self, x_min, x_max, target_wavelength):
        """
        Plot group index for multiple devices along with simulated data.

        Returns:
        - None
        """
        # Simulated data coefficients for 1550nm
        ng_500nm_fit_1550 = [-4.58401408e-07, 7.63213215e-05, -1.90478033e-03, 4.11962711e+00]
        ng_wavl_fit_1550 = [7.71116919e-16, 8.52254170e-13, 1.12019032e-09, 1.45999992e-06]

        # Simulated data coefficients for 1310nm
        ng_500nm_fit_1310 = [-6.78194041e-10, -1.83117238e-08, 3.25911055e-04, 4.40335210e+00]
        ng_wavl_fit_1310 = [7.56617044e-17, 1.84233132e-13, 4.86069882e-10, 1.28000000e-06]

        if self.label == 1310:
            ng_500nm_fit = ng_500nm_fit_1310
            ng_wavl_fit = ng_wavl_fit_1310
            title = '350 nm'
        elif self.label == 1550:
            ng_500nm_fit = ng_500nm_fit_1550
            ng_wavl_fit = ng_wavl_fit_1550
            title = '500 nm'
        else:
            ng_500nm_fit = None
            ng_wavl_fit = None
            title = None
            print("Label not specified")

        # Simulated data in meters
        ng_500nm = np.polyval(ng_500nm_fit, np.linspace(0, 100 - 1, 100))
        wavl_sim = np.polyval(ng_wavl_fit, np.linspace(0, 100 - 1, 100)) * 1e9  # Convert to nanometers

        # Apply truncation to simulation data with conversion to meters
        if x_min and x_max:
            range_min_meters = x_min * 1e-9  # Convert nm to meters
            range_max_meters = x_max * 1e-9  # Convert nm to meters
            wavl_sim, ng_500nm = siap.analysis.truncate_data(
                wavl_sim, ng_500nm, range_min_meters, range_max_meters
            )

        fig, ax1 = plt.subplots(figsize=(10, 6))

        cleaned_wavl = []
        cleaned_ng = []

        for device in self.devices:
            # Remove outliers for the current device
            cleaned_wavl_device, cleaned_ng_device = self.remove_outliers(device.ng_wavl, device.ng)
            cleaned_wavl.append(cleaned_wavl_device)
            cleaned_ng.append(cleaned_ng_device)

            # Scatter plot for the cleaned data
            ax1.scatter(cleaned_wavl_device, cleaned_ng_device, color='black', linewidth=0.1)

        if len(cleaned_wavl) == 0 or len(cleaned_ng) == 0:
            print("Warning: No valid data available for plotting.")
            return None, None

        # Determine the common x-axis (wavelength range) for averaging
        common_wavl = np.linspace(self.wavl_range[0], self.wavl_range[1])

        # Calculate the average and standard deviation
        ng_avg_wavl, ng_avg, ng_std, ng_std_avg = self.average_arrays(cleaned_wavl, cleaned_ng, common_wavl)

        target_wavelength = np.asarray(target_wavelength, dtype=ng_avg_wavl.dtype)
        idx = np.argmin(np.abs(ng_avg_wavl - target_wavelength))

        print(
            f"Group Index at {target_wavelength} nm is {ng_avg[idx]} ± {ng_std[idx]} for {self.name}_{self.pol}{self.wavl}")

        gindex = ng_avg[idx]
        gindexError = ng_std[idx]

        # Plot the average and standard deviation
        ax1.plot(ng_avg_wavl, ng_avg, '--', color='black', label='Average')
        ax1.fill_between(ng_avg_wavl, ng_std_avg - ng_std, ng_std_avg + ng_std, color='gray', alpha=0.2,
                         label='Std dev')

        ax1.set_xlim(np.min([np.min(wavl) for wavl in cleaned_wavl]), np.max([np.max(wavl) for wavl in cleaned_wavl]))
        ax1.plot(wavl_sim, ng_500nm, color='blue', label=f'Simulated {title} X 220 nm')
        ax1.legend()
        ax1.set_ylabel('Group index')
        ax1.set_xlabel('Wavelength [nm]')

        pdf_path_gindex, pdf_path_contour = self.saveGraph()
        plt.savefig(pdf_path_gindex, format='pdf')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_gsim', 'Figure': img_buffer},
            ignore_index=True
        )

        return gindex, gindexError

    def sort_devices_by_length(self):
        self.devices = sorted(self.devices, key=lambda d: d.length)

    def plot_coupling_coefficient_contour(self):
        """
        Plot a contour map of coupling coefficients.

        Returns:
        - None
        """
        ng_wavl = [device.ng_wavl[:11] for device in self.devices]
        device_lengths = [device.length for device in self.devices]

        # Creating a common wavelength grid
        common_ng_wavl = np.unique(np.concatenate(ng_wavl))
        common_ng_wavl.sort()
        # Creating meshgrid for wavelength and device length
        X, Y = np.meshgrid(common_ng_wavl, device_lengths)
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

        contour = ax.contourf(X, Y, Z, cmap='viridis')
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Coupling Length [µm]")
        ax.set_title("Coupling Coefficient Contour Map (100 nm gap)")
        fig.colorbar(contour)

        pdf_path_gindex, pdf_path_contour = self.saveGraph()
        plt.savefig(pdf_path_contour, format='pdf')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        self.df_figures = self.df_figures._append(
            {'Name': f'{self.name}_{self.pol}{self.wavl}_coup', 'Figure': img_buffer},
            ignore_index=True
        )
        # plt.show()  # Display the plot

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
        output_directory = os.path.join(self.main_script_directory, f"{self.name}_{self.pol}{self.wavl}")
        os.makedirs(output_directory, exist_ok=True)
        pdf_path_gindex = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_gindex.pdf")
        pdf_path_contour = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}_contour.pdf")

        return pdf_path_gindex, pdf_path_contour

    def remove_outliers(self, data_x, data_y, threshold=3):
        """
        Removes outliers from data based on their deviation from the median absolute deviation (MAD).

        :param data_x: Numpy array of x-axis data
        :param data_y: Numpy array of y-axis data
        :param threshold: Number of median absolute deviations (MAD) from the median to consider as outliers
        :return: Numpy arrays with outliers removed, average, and standard deviation of the cleaned data
        """
        combined_data = np.column_stack((data_x, data_y))

        median = np.median(combined_data[:, 1])
        mad = np.median(np.abs(combined_data[:, 1] - median))

        # Define lower and upper bounds for outliers
        lower_bound = median - threshold * mad
        upper_bound = median + threshold * mad
        cleaned_data = combined_data[(combined_data[:, 1] >= lower_bound) & (combined_data[:, 1] <= upper_bound)]

        cleaned_x = cleaned_data[:, 0]
        cleaned_y = cleaned_data[:, 1]

        return cleaned_x, cleaned_y