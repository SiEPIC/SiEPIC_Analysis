
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import siepic_analysis_package as siap

import pandas as pd

import io
import re
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class Device:
    def __init__(self, wavl, pol, root_path, main_script_directory, files_path, target_prefix, target_suffix, port,
                 name, characterization):
        self.wavl = wavl
        self.pol = pol
        self.root_path = root_path
        self.main_script_directory = main_script_directory
        self.files_path = files_path
        self.target_prefix = target_prefix
        self.target_suffix = target_suffix
        self.port = port
        self.name = name
        self.characterization = characterization
        self.figures_df = pd.DataFrame(columns=['Name', 'Figure'])

    def get_waveguide_length(self, device_id):
        """
        Extract the waveguide length from a device ID by removing specified prefixes and suffixes.

        Args:
        device_id (str): The device ID from which to extract the waveguide length.

        Returns:
        float: The waveguide length extracted from the device ID.
        """
        try:
            start_index = device_id.index(self.target_prefix) + len(self.target_prefix)

            if self.target_suffix:  # If suffix is not empty
                end_index = device_id.index(self.target_suffix, start_index)
                value_str = device_id[start_index:end_index]
            else:
                # If suffix is empty, extract all digits and optionally decimal points
                value_str = re.findall(r'[0-9.]+', device_id[start_index:])[0]

            device_value = float(value_str)
            return device_value
        except (ValueError, IndexError):
            # Handle cases where the prefix, suffix, or device_value is not found
            return None

    def loadData(self):
        """
        Load data from files into dataframes.

        Args:
            files_path (str): Path to the directory containing the data files.
            target_prefix (str): Prefix of the files to be loaded.
            target_suffix (str): Suffix of the files to be loaded.

        Returns:
            wavelengths_file (list): List of wavelengths.
            channel_pwr (list): List of dataframes containing data of power(dBm) measured.
        """
        wavelengths_file = []
        channel_pwr = []

        for root, dirs, files in os.walk(self.files_path):
            if os.path.basename(root).startswith(self.target_prefix):
                for file in files:
                    if file.endswith(".csv"):
                        channel = siap.analysis.processCSV(root + r'/' + file)
                        channel_pwr.append(channel)
                        wavelengths_file.append(self.get_waveguide_length(channel.deviceID))
        return wavelengths_file, channel_pwr

    def process_data(self, wavelengths_file, channel_pwr):
        """
        Process and transform wavelength and power data for analysis.

        This method takes wavelength and power data as input and processes it based on the characterization
        type specified by `self.characterization`. For 'cutback_waveguide', it converts the wavelength from
        nanometers to centimeters and sorts it. For 'cutback_device', it sorts the wavelength data without
        conversion. It also creates an input array suitable for analysis.

        Parameters:
        - wavelengths_file (list): A list of wavelengths in nanometers.
        - channel_pwr (list): A list of power data for different channels.

        Returns:
        - lengths_cm (list): Processed wavelength data in centimeters (for 'cutback_waveguide').
        - lengths_cm_sorted (list): Sorted and processed wavelength data in centimeters.
        - lengths_um (list): Unprocessed wavelength data in nanometers.
        - input_to_function (list): An array of wavelength and power data for analysis.
        """
        if self.characterization == 'Insertion Loss (dB/cm)':
            lengths_cm = [i / 10000 for i in wavelengths_file]
        elif self.characterization == 'Insertion Loss (dB/device)':
            lengths_cm = [i for i in wavelengths_file]

        lengths_cm_sorted = sorted(lengths_cm)
        lengths_um = wavelengths_file

        input_to_function = []

        for channel in channel_pwr:
            input_to_function.append([np.array(channel.wavl), np.array(channel.pwr[self.port])])

        return lengths_cm, lengths_cm_sorted, lengths_um, input_to_function

    def getSets(self, input_to_function, lengths_um, wavl_min=None, wavl_max=None):
        """
        Separate input data into dictionaries, sorted by length, truncating wavelengths and powers based on wavl_min and wavl_max.

        Args:
        input_to_function (list): Input data containing wavelength and power arrays.
        lengths_um (list): A list of lengths in micrometers.
        wavl_min (float or None): Minimum wavelength to truncate data. Default is None.
        wavl_max (float or None): Maximum wavelength to truncate data. Default is None.

        Returns:
        dict: A dictionary where keys are lengths (in string format) and values are dictionaries
        containing truncated wavelength and power arrays.
        """
        sorted_data = []
        for i in range(0, len(input_to_function)):
            key = lengths_um[i]
            wavelength = input_to_function[i][0]
            power = input_to_function[i][1]

            if wavl_min is not None or wavl_max is not None:
                # Truncate wavelengths and corresponding powers based on wavl_min and wavl_max
                indices = []
                if wavl_min is not None:
                    indices = [idx for idx, wv in enumerate(wavelength) if wv >= wavl_min]
                if wavl_max is not None:
                    indices = [idx for idx in indices if wavelength[idx] <= wavl_max]

                wavelength = wavelength[indices]
                power = power[indices]

            data = {
                "wavelength": wavelength,
                "power": power
            }
            sorted_data.append((key, data))

        sorted_data.sort()
        data_sets = {str(key): data for key, data in sorted_data}

        return data_sets

    def graphRaw(self, separated_data):
        """
        Create a combined graph for raw data with automatically defined line colors.

        Args:
            separated_data (dict): A dictionary containing separated data sets.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        cmap = cm.get_cmap("tab10")

        for i, (key, data) in enumerate(separated_data.items()):
            color = cmap(i % 10)

            if self.characterization == 'cutback_waveguide':
                label = f"L = {key}um"
            elif self.characterization == 'cutback_device':
                label = f"Number of Devices = {key}"
            else:
                label = str(key)

            plt.plot(
                data["wavelength"],
                data["power"],
                label=label,
                color=color
            )

        plt.ylabel('Power (dBm)', color='black')
        plt.xlabel('Wavelength (nm)', color='black')
        plt.title(f"Raw Measurement of Cutback Structures for {self.name}_{self.pol}{self.wavl}nm")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})
        plt.legend()

        pdf_path_raw, pdf_path_cutback = self.saveGraph()
        plt.savefig(pdf_path_raw, format='pdf')
        # plt.show()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        df_figures = pd.DataFrame([{'Name': f'{self.name}_{self.pol}{self.wavl}_1', 'Figure': img_buffer}])
        return {'Name': f'{self.name}_{self.pol}{self.wavl}_1', 'Figure': img_buffer}, df_figures

    def getArrays(self, separated_data):
        """
        Extract power arrays and wavelength data from the input data.

        Args:
        input_to_function (dict): Separated data containing wavelengths and power arrays.
        lengths_um (list): A list of lengths in micrometers.

        Returns:
        tuple: A tuple containing power arrays and wavelength data.
        power_arrays (list): A list of power arrays, one for each length.
        wavelength_data (array): Wavelength data common to all power arrays.
        """
        power_arrays = []
        wavelength_data = None
        for i, (key, data) in enumerate(separated_data.items()):
            power_arrays.append(data['power'])
            if i == 0:
                wavelength_data = data['wavelength']

        if wavelength_data is None:
            print("No wavelength data found!")

        return power_arrays, wavelength_data

    def getSlopes(self, input_data, lengths_cm_sorted):
        """
        Calculate slopes for the given input data.

        Args:
        input_data (list): List of power arrays.
        lengths_cm_sorted (list): Sorted list of lengths in centimeters.
        wavelength_data (list): List of wavelength data corresponding to the power arrays.

        Returns:
        list: List of calculated slopes.
        """
        slopes = []

        num_entries = len(input_data[0])
        for i in range(num_entries):
            x_values = lengths_cm_sorted
            y_values = []

            for power_array in input_data:
                y_values.append(power_array[i])
            coefficients = np.polyfit(x_values, y_values, 1)  # Using np.polyfit with degree 1 for linear fit

            slope = coefficients[0]
            slopes.append(slope)

        return slopes

    def graphCutback(self, wavl, wavelength_data, slopes, degree=3):
        """
        Generate a graph of wavelength vs. cutback loss along with a line of best fit.

        Args:
        wavl (int): Wavelength value.
        wavelength_data (array): Array of wavelength data.
        slopes (array): Array of slopes.

        Returns:
        float: The cutback loss at the specified target wavelength.
        """
        slopes = np.array(slopes)
        valid_indices = np.where(~np.isnan(slopes))[0]

        filtered_wavelength_data = wavelength_data[valid_indices]
        filtered_slopes = slopes[valid_indices]

        coefficients3 = np.polyfit(filtered_wavelength_data, filtered_slopes, degree)
        poly_func = np.poly1d(coefficients3)
        x_fit = np.linspace(filtered_wavelength_data.min(), filtered_wavelength_data.max(), 1000)
        y_fit = poly_func(x_fit)

        target_wavelength = wavl
        slope_at_wavl = -(poly_func(target_wavelength))
        error = np.abs(filtered_slopes - poly_func(filtered_wavelength_data))

        plt.figure(figsize=(10, 6))
        plt.plot(filtered_wavelength_data, np.abs(filtered_slopes), color='blue', marker='', linestyle='-',
                 label='Insertion loss (raw)')
        plt.plot(x_fit, np.abs(y_fit), color='red', linestyle='-', label='Insertion loss (fit)', linewidth=3)
        plt.xlabel('Wavelength (nm)', color='black')

        if self.characterization == 'cutback_waveguide':
            plt.ylabel('Insertion Loss (dB/cm)', color='black')
        elif self.characterization == 'cutback_device':
            plt.ylabel('Insertion Loss (dB/device)', color='black')

        plt.title(
            f"Insertion Losses Using the Cutback Method for {self.name}_{self.pol}{self.wavl}nm")
        plt.grid(True)
        plt.legend()
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        print(
            f'The insertion loss at wavelength = {target_wavelength} is {slope_at_wavl} +/- {error[filtered_wavelength_data == target_wavelength][0]} for {self.name}_{self.pol}{self.wavl}')  # Updated naming

        cutback_error = error[filtered_wavelength_data == target_wavelength][0]

        pdf_path_raw, pdf_path_cutback = self.saveGraph()
        plt.savefig(pdf_path_cutback, format='pdf')

        # plt.show()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        df_figures = pd.DataFrame([{'Name': f'{self.name}_{self.pol}{self.wavl}_2', 'Figure': img_buffer}])
        self.figures_df = pd.concat([self.figures_df, df_figures], ignore_index=True)

        return slope_at_wavl, cutback_error, df_figures

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

        pdf_path_raw = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}nm_raw.pdf")
        pdf_path_cutback = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}nm_cutback.pdf")

        return pdf_path_raw, pdf_path_cutback

    def graphCutback_CDC(self, wavelength_min, wavelength_max, wavelength_data, slopes, standard_error):
        """
        Generate a graph of wavelength vs. cutback loss.

        Args:
        wavelength_min (int): Minimum wavelength value.
        wavelength_max (int): Maximum wavelength value.
        wavelength_data (array): Array of wavelength data.
        slopes (array): Array of slopes.

        Returns:
        tuple: The cutback loss at the specified target wavelength, error, and dataframe with the plot.
        """
        slopes = np.array(slopes)
        valid_indices = np.where(~np.isnan(slopes))[0]

        filtered_wavelength_data = wavelength_data[valid_indices]
        filtered_slopes = slopes[valid_indices]

        # Define the target wavelength
        target_wavelength = (wavelength_min + wavelength_max) / 2

        # Find the closest original data point to the target wavelength
        closest_index = np.argmin(np.abs(filtered_wavelength_data - target_wavelength))
        slope_at_target = -(filtered_slopes[closest_index])

        # Plotting the raw data and target wavelength
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_wavelength_data, np.abs(filtered_slopes), 'o-', color='blue', label='Insertion Loss (raw)')
        plt.axvline(x=target_wavelength, color='g', linestyle='--', label=f'Target Wavelength ({target_wavelength} nm)')
        plt.ylabel('Insertion Loss (dB/device)', color='black')
        plt.xlabel('Wavelength (nm)', color='black')
        plt.title(f"Insertion Losses Using the Cutback Method for {self.name}_{self.pol}{self.wavl}nm")
        plt.grid(True)
        plt.legend()
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        error_at_target = standard_error.round(2)
        print(
            f'The insertion loss at wavelength = {target_wavelength} is {slope_at_target} +/- {error_at_target} for {self.name}_{self.pol}{self.wavl}')  # Updated naming

        # Save the plot
        pdf_path_raw, pdf_path_cutback = self.saveGraph()  # Custom method to save graphs
        plt.savefig(pdf_path_cutback, format='pdf')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        df_figures = pd.DataFrame([{'Name': f'{self.name}_{self.pol}{self.wavl}_2', 'Figure': img_buffer}])
        self.figures_df = pd.concat([self.figures_df, df_figures], ignore_index=True)

        cutback_error = error_at_target

        return slope_at_target, cutback_error, df_figures

    def getSlopeUncertainty(self, input_data, lengths_cm_sorted, target_wavelength):
        """
        Calculate the uncertainty of the slope at the target wavelength.

        Args:
        input_data (list): List of power arrays.
        lengths_cm_sorted (list): Sorted list of lengths in centimeters.
        target_wavelength (float): The wavelength at which to calculate the uncertainty.

        Returns:
        float: The standard error of the slope at the target wavelength.
        """
        standard_error = []
        standard_error_at_closest = None
        num_entries = len(input_data[0])
        for i in range(num_entries):
            x_values = lengths_cm_sorted
            y_values = []

            for power_array in input_data:
                y_values.append(power_array[i])
            coefficients = np.polyfit(x_values, y_values, 1)  # Using np.polyfit with degree 1 for linear fit

            # Calculate residuals
            residuals = np.array(y_values) - (coefficients[0] * np.array(x_values) + coefficients[1])

            # Calculate variance of residuals
            var_residuals = np.var(residuals, ddof=2)  # ddof=2 for unbiased estimate

            # Calculate standard error of the slope
            std_error_slope = np.sqrt(var_residuals) / np.sqrt(np.sum((np.array(x_values) - np.mean(x_values)) ** 2))
            standard_error.append(std_error_slope)

            # Find the index of the closest wavelength to the target wavelength
            closest_index = np.argmin(np.abs(np.array(x_values) - target_wavelength))

            # Get the standard error at the closest wavelength
            standard_error_at_closest = standard_error[closest_index] if closest_index < len(standard_error) else None

        return standard_error_at_closest