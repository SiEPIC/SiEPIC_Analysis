import os
import yaml
from analysis_packageV2 import Device

class Execute:
    def __init__(self, root_path):
        self.root_path = root_path

    def load_and_analyze(self):
        results_directory = os.path.join(self.root_path, "analysis_results")

        # Check if the "analysis_results" folder exists, and create it if it doesn't
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # Construct the path to your .yaml file using root_path
        yaml_file = os.path.join(self.root_path, 'config.yaml')

        # Load data from the .yaml file
        with open(yaml_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        # Iterate through the data sets and perform the analysis
        for dataset in data['devices']:
            name = dataset['name']
            wavl = dataset['wavelength']
            pol = dataset['polarization']
            files_path = os.path.join(self.root_path, f"{wavl}_{pol}")
            target_prefix = dataset['target_prefix']
            target_suffix = dataset['target_suffix']
            characterization = dataset['characterization']
            port = dataset['port']

            # Create an instance of the Device class (Assuming you have a Device class)
            device = Device(wavl, self.root_path, results_directory, files_path, target_prefix, target_suffix, port, name, characterization)

            # Call the execute method to perform the analysis
            device.execute(target_wavelength=wavl)  # You can specify the target_wavelength if needed
