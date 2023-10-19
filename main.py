# import os
import yaml
from analysis_packageV2 import Device

if __name__ == "__main__":
    import os

    root_path = "D:\\Academics\\PyCharmProjects\\Data\\Actives-May-2023"
    wavetype = 'TE'
    # Load data from the YAML file
    with open('config.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # Iterate through the data sets and perform the analysis
    for dataset in data['devices']:
        wavl = dataset['wavelength']
        output_path_cutback = dataset['output_path_cutback']
        output_path_raw = dataset['output_path_raw']
        files_path = os.path.join(root_path, f"{wavl}_TE")
        target_prefix = dataset['target_prefix']
        target_suffix = dataset['target_suffix']
        port = dataset['port']

        # Create an instance of the Device class
        device = Device(wavl, root_path, output_path_cutback, output_path_raw, files_path, target_prefix, target_suffix, port)

        # Call the execute method to perform the analysis
        device.execute(target_wavelength=wavl)  # You can specify the target_wavelength if needed


"""
# Usage:
if __name__ == "__main__":
    # wavl = 1550
    root_path = "D:\\Academics\\PyCharmProjects\\Data\\Actives-May-2023"

    # output_path_cutback = 'D:\\Downloads\\output_cutback.pdf'
    # output_path_raw = 'D:\\Downloads\\output_raw.pdf'
    # target_prefix = "PCM_SpiralWG"
    # target_suffix = "TE"
    # port = 1

    # Load data from the YAML file
    with open('config.yaml', 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # Iterate through the data sets and perform the analysis
    for dataset in data:
        wavl = dataset['wavelength']
        output_path_cutback = dataset['output_path_cutback']
        output_path_raw = dataset['output_path_raw']
        files_path = os.path.join(root_path, f"{wavl}_TE")
        target_prefix = dataset['target_prefix']
        target_suffix = dataset['target_suffix']
        port = dataset['port']

        # Create an instance of the Device class
        device = Device(wavl, root_path, output_path_cutback, output_path_raw, files_path, target_prefix, target_suffix,
                        port)

        # Call the execute method to perform the analysis
        device.execute(target_wavelength=wavl)  # You can specify the target_wavelength if needed

    # files_path = os.path.join(root_path, f"{wavl}_TE")

    # Create an instance of the Device class
    # device = Device(wavl, root_path, output_path_cutback, output_path_raw, files_path, target_prefix, target_suffix,
                    # port)

    # Call the execute function to perform the steps
    # device.execute(wavl)
"""