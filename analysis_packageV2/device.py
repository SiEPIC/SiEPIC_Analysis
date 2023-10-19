import sys
sys.path.append(r'D:\Academics\PyCharmProjects')  # Add the directory to sys.path
import siepic_analysis_package as siap
import os  # Import the os module
import numpy as np  # You may need to import other modules as well
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class Device:
    def __init__(self, wavl, root_path, output_path_cutback, output_path_raw, files_path, target_prefix, target_suffix, port):
        self.wavl = wavl
        self.root_path = root_path
        self.output_path_cutback = output_path_cutback
        self.output_path_raw = output_path_raw
        self.files_path = os.path.join(root_path, f"{wavl}_TE")  # Define files_path as an instance variable
        self.target_prefix = target_prefix
        self.target_suffix = target_suffix
        self.port = port

    def get_waveguide_length(self, device_id):
        """
        Extract the waveguide length from a device ID by removing specified prefixes and suffixes.

        Args:
        device_id (str): The device ID from which to extract the waveguide length.

        Returns:
        float: The waveguide length extracted from the device ID.
        """
        return float(device_id.removeprefix(self.target_prefix).removesuffix(self.target_suffix))

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

        print("Entering os.walk loop")
        for root, dirs, files in os.walk(self.files_path):
            print(f"level0 Root: {root}")
            print(f"level0 Directories: {dirs}")
            print(f"level0 Files: {files}")

            print("level1 target_prefix:", self.target_prefix)
            print("level1 basename root:", os.path.basename(root))
            if os.path.basename(root).startswith(self.target_prefix):
                print('level2')
                for file in files:
                    print('level3')
                    if file.endswith(".csv"):
                        print('level4')

                        channel = siap.analysis.processCSV(os.path.join(root, file))
                        channel_pwr.append(channel)

                        print("level4 channel:", channel)
                        print("level4 channel power:", channel_pwr.append(channel))

                        wavelengths_file.append(self.get_waveguide_length(channel.deviceID))

                        print("level4 channel.name:", channel.deviceID)

        return wavelengths_file, channel_pwr

    def process_data(self, wavelengths_file, channel_pwr):
        # Divide by 10000 to see the result in dB/cm
        lengths_cm = [i / 10000 for i in wavelengths_file]

        # if waveguide ...

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
        for i in range(0, len(input_to_function), num_arrays_per_dict):
            key = lengths_um[i // num_arrays_per_dict]
            data = {
                "wavelength": input_to_function[i][0],
                "power": input_to_function[i + 1][1]
            }
            sorted_data.append((key, data))

        # Sort the list of tuples by the keys (lengths_um)
        sorted_data.sort()

        # Create the final data_sets dictionary with sorted data
        data_sets = {str(key): data for key, data in sorted_data}

        return data_sets

    def graphRaw(self, separated_data, output_path_raw):
        """
        Create a combined graph for raw data with automatically defined line colors.

        Args:
            separated_data (dict): A dictionary containing separated data sets.
            output_path_raw (str): Path to save the graph.

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
            plt.plot(
                data["wavelength"],
                data["power"],
                label=f"L = {key}um",
                color=color
            )

        plt.ylabel('Power (dBm)', color='black')
        plt.xlabel('Wavelength (nm)', color='black')
        plt.title("Raw Measurement of Cutback Structures")
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})
        plt.legend()  # Display legends for different sets

        plt.savefig(output_path_raw, format='pdf')
        plt.show()  # Display the combined graph

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
            print(f"Key: {key}")
            print(f"Power: {data['power']}")
            print()  # Just for separating the sets visually

            # Store the power array in the power_arrays list
            power_arrays.append(data['power'])

            # Store the wavelength data from the first file as a 1D array
            if i == 0:
                wavelength_data = data['wavelength']

        # Now, power_arrays will contain all the power arrays
        print("All Power Arrays:")
        for i, power_array in enumerate(power_arrays):
            print(f"Power Array {i + 1}: {power_array}")

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

            # Check if the current iteration corresponds to 1310 nm
            if i == index_target:
                x_target = x_values
                y_target = y_values

        # Print the number of slopes calculated
        print(f"Number of slopes calculated: {len(slopes)}")

        # Print the slope at 1310 nm
        print(f"Slope at {target_wavelength} nm: {slopes[index_target]}")

        # Print x and y data at 1310 nm
        print(f"x data at {target_wavelength} nm: {x_target}")
        print(f"y data at {target_wavelength} nm: {y_target}")

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

        # plt.ylabel('Propagation Loss (dB/cm)', color='black')
        plt.ylabel('Insertion Loss (dB/cm)', color='black')
        plt.xlabel('Wavelength (nm)', color='black')
        plt.title("Insertion Losses Using the Cutback Method")
        plt.grid(True)
        plt.legend()
        matplotlib.rcParams.update({'font.size': 11, 'font.family': 'Times New Roman', 'font.weight': 'bold'})

        # Print the cutback loss at the target wavelength
        print(
            f'The insertion loss at wavelength = {target_wavelength} is {slope_at_wavl} +/- {error[wavelength_data == target_wavelength][0]}')

        plt.savefig(self.output_path_cutback, format='pdf')

        # Show the plot
        plt.show()

        return slope_at_wavl

    def execute(self, target_wavelength=None):
        # Load data
        wavelengths_file, channel_pwr = self.loadData()

        # Process data
        lengths_cm, lengths_cm_sorted, lengths_um, input_to_function = self.process_data(wavelengths_file, channel_pwr)

        # Print channel_pwr and wavelengths_file
        print(f'channel_pwr is: {channel_pwr}')
        print(f'wavelengths_file is: {wavelengths_file}')
        print(f'length of input_to_function is: {len(input_to_function)}')

        # Call the getSets method
        separated_data = self.getSets(input_to_function, lengths_um)

        # Print the separated data for verification
        for key, data in separated_data.items():
            print(f"Key: {key}")
            print(f"Wavelength: {data['wavelength']}")
            print(f"Power: {data['power']}")
            print()

        # Call the graphRaw method on the device object
        self.graphRaw(separated_data, self.output_path_raw)

        # Call the getArrays method
        power_arrays, wavelength_data = self.getArrays(input_to_function, lengths_um)

        # Call the getSlopes method
        if target_wavelength is not None:
            slopes = self.getSlopes(power_arrays, lengths_cm_sorted, wavelength_data, target_wavelength)

            # Call the graphCutback method
            cutback_loss = self.graphCutback(self.wavl, wavelength_data, slopes)
        else:
            print("Target wavelength not specified. Skipping getSlopes and graphCutback.")