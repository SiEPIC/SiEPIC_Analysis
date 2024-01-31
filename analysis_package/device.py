import io
import sys

import pandas as pd

sys.path.append(r'D:\Academics\PyCharmProjects')  # Add the directory to sys.path
import siepic_analysis_package as siap
import os  # Import the os module
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class Device:
    def __init__(self, wavl, pol, root_path, main_script_directory, files_path, target_prefix, target_suffix, port, name, characterization):
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
            end_index = device_id.index(self.target_suffix, start_index)
            device_value = float(device_id[start_index:end_index])
            return device_value

        except ValueError:
            # Handle the case where prefix or suffix is not found
            return None  # Return None to indicate failure
            
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
                        channel = siap.analysis.processCSV(root+r'/'+file)
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
            # Divide by 10000 to see the result in dB/cm
            lengths_cm = [i / 10000 for i in wavelengths_file]
        elif self.characterization == 'Insertion Loss (dB/device)':
            lengths_cm = [i for i in wavelengths_file]

        # Sort lengths_cm from smallest to largest
        lengths_cm_sorted = sorted(lengths_cm)

        lengths_um = wavelengths_file

        input_to_function = []

        for channel in channel_pwr:
            input_to_function.append([np.array(channel.wavl), np.array(channel.pwr[self.port])])

        return lengths_cm, lengths_cm_sorted, lengths_um, input_to_function

    def getSets(self, input_to_function, lengths_um, num_arrays_per_dict=2):
        """
        Separate input data into dictionaries, sorted by length.

        Args:
        input_to_function (list): Input data containing wavelength and power arrays.
        lengths_um (list): A list of lengths in micrometers.
        num_arrays_per_dict (int): Number of arrays (wavelength and power) per dictionary. Default is 2.

        Returns:
        dict: A dictionary where keys are lengths (in string format) and values are dictionaries
        containing wavelength and power arrays.
        """
        # Initialize a list to store tuples of (key, data)
        sorted_data = []

        # Separate the arrays into dictionaries
        for i in range(0, len(input_to_function)):
            key = lengths_um[i]
            data = {
                "wavelength": input_to_function[i][0],
                "power": input_to_function[i][1]
            }
            sorted_data.append((key, data))

        # Sort the list of tuples by the keys (lengths_um)
        sorted_data.sort()

        # Create the final data_sets dictionary with sorted data
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
        # Create a new figure for the combined graph
        plt.figure(figsize=(10, 6))

        # Get a colormap to automatically define line colors
        cmap = cm.get_cmap("tab10")

        # Plot each dataset with a different line color
        for i, (key, data) in enumerate(separated_data.items()):
            color = cmap(i % 10)  # Use modulo to cycle through the colormap

            if self.characterization == 'cutback_waveguide':
                label = f"L = {key}um"
            elif self.characterization == 'cutback_device':
                label = f"Number of Devices = {key}"
            else:
                label = str(key)  # Default label

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
        plt.legend()  # Display legends for different sets

        pdf_path_raw, pdf_path_cutback = self.saveGraph()
        plt.savefig(pdf_path_raw, format='pdf')
        # plt.show()  # Display the combined graph

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Convert the BytesIO object to a DataFrame
        df_figures = pd.DataFrame([{'Name': f'{self.name}_{self.pol}{self.wavl}_1', 'Figure': img_buffer}])

        # Return a dictionary containing the raw_cutback_img and other information
        return {'Name': f'{self.name}_{self.pol}{self.wavl}_1', 'Figure': img_buffer}, df_figures

    def getArrays(self, input_to_function, lengths_um):
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
        # Get the separated data
        separated_data = self.getSets(input_to_function, lengths_um)

        power_arrays = []
        wavelength_data = None  # Initialize as None

        # Print the separated data for verification and store power arrays
        for i, (key, data) in enumerate(separated_data.items()):
            # Store the power array in the power_arrays list
            power_arrays.append(data['power'])

            # Store the wavelength data from the first file as a 1D array
            if i == 0:
                wavelength_data = data['wavelength']

        # Check if wavelength_data is None (no data found in any file)
        if wavelength_data is None:
            print("No wavelength data found!")

        return power_arrays, wavelength_data

    def getSlopes(self, input_data, lengths_cm_sorted, wavelength_data, target_wavelength):
        """
        Calculate slopes for the given input data.

        Args:
        input_data (list): List of power arrays.
        lengths_cm_sorted (list): Sorted list of lengths in centimeters.
        wavelength_data (list): List of wavelength data corresponding to the power arrays.

        Returns:
        list: List of calculated slopes.
        """
        # Create an empty list to store the slopes
        slopes = []

        # Find the index corresponding to target_wavelength
        index_target = np.argmin(np.abs(np.array(wavelength_data) - target_wavelength))

        # Initialize empty lists to store x and y data at 1310 nm
        x_target = []
        y_target = []

        # Iterate through all the entries in the power arrays
        num_entries = len(input_data[0])
        for i in range(num_entries):
            # Initialize lists to store x and y values for the current entry
            x_values = lengths_cm_sorted
            y_values = []

            # Iterate through the power arrays
            for power_array in input_data:
                y_values.append(power_array[i])

            # Calculate the coefficients for a linear polynomial fit
            coefficients = np.polyfit(x_values, y_values, 1)  # Using np.polyfit with degree 1 for linear fit

            # The first coefficient (coefficients[0]) represents the slope of the linear fit
            slope = coefficients[0]

            # Append the calculated slope to the list of slopes
            slopes.append(slope)

            # Check if the current iteration corresponds to target wavelength
            # if i == index_target:
                # x_target = x_values
                # y_target = y_values

        # Print the slope at target wavelength
        # print(f"Slope at {target_wavelength} nm: {slopes[index_target]}")

        # Print x and y data at target wavelength
        # print(f"x data at {target_wavelength} nm: {x_target}")
        # print(f"y data at {target_wavelength} nm: {y_target}")

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
        # Fit a polynomial of degree 3 to your data
        coefficients3 = np.polyfit(wavelength_data, slopes, degree)

        # Create a polynomial function from the coefficients
        poly_func = np.poly1d(coefficients3)

        # Generate x values for the line of best fit
        x_fit = np.linspace(wavelength_data.min(), wavelength_data.max(), 1000)

        # Calculate the corresponding y values using the polynomial function
        y_fit = poly_func(x_fit)

        # Calculate the cutback loss at the target wavelength
        target_wavelength = wavl
        slope_at_wavl = np.abs(poly_func(target_wavelength))

        # Calculate the error bars (+/-)
        error = np.abs(slopes - poly_func(wavelength_data))

        # Create a plot of wavelength vs. slope
        plt.figure(figsize=(10, 6))
        plt.plot(wavelength_data, np.abs(slopes), color='blue', marker='', linestyle='-', label='Insertion loss (raw)')
        plt.plot(x_fit, np.abs(y_fit), color='red', linestyle='-', label='Insertion loss (fit)', linewidth=3)

        # Plot labels
        plt.xlabel('Wavelength (nm)', color='black')

        if self.characterization == 'cutback_waveguide':
            plt.ylabel('Insertion Loss (dB/cm)', color='black')
        elif self.characterization == 'cutback_device':
            plt.ylabel('Insertion Loss (dB/device)', color='black')

        plt.title(
            f"Insertion Losses Using the Cutback Method for {self.name}_{self.pol}{self.wavl}nm")  # Updated naming
        plt.grid(True)
        plt.legend()
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        # Print the cutback loss at the target wavelength
        print(
            f'The insertion loss at wavelength = {target_wavelength} is {slope_at_wavl} +/- {error[wavelength_data == target_wavelength][0]} for {self.name}_{self.pol}{self.wavl}')  # Updated naming

        cutback_error = error[wavelength_data == target_wavelength][0]

        pdf_path_raw, pdf_path_cutback = self.saveGraph()
        plt.savefig(pdf_path_cutback, format='pdf')

        # Show the plot
        # plt.show()

        # Save the Matplotlib figure to a BytesIO object
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Convert the BytesIO object to a DataFrame
        df_figures = pd.DataFrame([{'Name': f'{self.name}_{self.pol}{self.wavl}_2', 'Figure': img_buffer}])

        # Add the figure to the existing figures DataFrame
        self.figures_df = pd.concat([self.figures_df, df_figures], ignore_index=True)

        # Return a dictionary containing the cutback loss and other information
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
        # Create a directory based on self.name, self.pol, and self.wavl if it doesn't exist
        output_directory = os.path.join(self.main_script_directory, f"{self.name}_{self.pol}{self.wavl}")
        os.makedirs(output_directory, exist_ok=True)

        # Combine the directory and the filename to get the full paths
        pdf_path_raw = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}nm_raw.pdf")
        pdf_path_cutback = os.path.join(output_directory, f"{self.name}_{self.pol}{self.wavl}nm_cutback.pdf")

        # Now, you can save your PDFs to pdf_path_raw and pdf_path_cutback
        return pdf_path_raw, pdf_path_cutback
