import os
import yaml
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime

from analysis_package import Device
from analysis_package.bragg import DirectionalCoupler
from analysis_package.groupIndex import GroupIndex
class Execute:
    def __init__(self, root_path):
        self.root_path = root_path
        self.results_list = []

    def analyze_cutback(self):
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
            cutback_value, cutback_error = device.execute(target_wavelength=wavl)

            # Append the results to the list
            result_entry = {'Name': name, 'Wavelength': wavl, 'Polarization': pol, 'Cutback Value': cutback_value,
                            'Error': cutback_error}
            # add error as val to return
            self.results_list.append(result_entry)

        # Convert the list of results to a DataFrame
        results_df = pd.DataFrame(self.results_list)

        # Round the 'Cutback Value' and 'Error' columns to two decimal places in the DataFrame
        results_df['Cutback Value'] = results_df['Cutback Value'].round(2)
        results_df['Error'] = results_df['Error'].round(2)

        self.pdfReport(results_df)

        return results_df

    def analyze_bragg(self):
        results_directory = os.path.join(self.root_path, "analysis_results")

        # Check if the "analysis_results" folder exists, and create it if it doesn't
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # Construct the path to your .yaml file using root_path
        yaml_file = os.path.join(self.root_path, 'braggconfig.yaml')

        # Load data from the .yaml file
        with open(yaml_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        # Iterate through the data sets and perform the analysis
        for dataset in data['devices']:
            name = dataset['name']
            wavl = dataset['wavelength']
            pol = dataset['polarization']
            files_path = os.path.join(self.root_path, f"{wavl}_{pol}")
            device_prefix = dataset['device_prefix']
            device_suffix = dataset['device_suffix']
            port_drop = dataset['port_drop']
            port_thru = dataset['port_thru']

            dc = DirectionalCoupler(
                fname_data=files_path,
                device_prefix=device_prefix,
                port_thru=port_thru,
                port_drop=port_drop,
                device_suffix=device_suffix,
                name=name,
                main_script_directory=results_directory
            )

            dc.process_files()
            dc.plot_devices()
            dc.plot_analysis_results()
            dc.overlay_simulation_data(target_wavelength=wavl)

    def analyze_gIndex(self):
        results_directory = os.path.join(self.root_path, "analysis_results")

        # Check if the "analysis_results" folder exists, and create it if it doesn't
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # Construct the path to your .yaml file using root_path
        yaml_file = os.path.join(self.root_path, 'groupIndexconfig.yaml')

        # Load data from the .yaml file
        with open(yaml_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        # Iterate through the data sets and perform the analysis
        for dataset in data['devices']:
            name = dataset['name']
            wavl = dataset['wavelength']
            pol = dataset['polarization']
            # files_path = os.path.join(self.root_path, f"{wavl}_{pol}")
            device_prefix = dataset['device_prefix']
            device_suffix = dataset['device_suffix']
            port_cross = dataset['port_cross']
            port_bar = dataset['port_bar']

            group_index = GroupIndex(directory_path=self.root_path,
                                     wavl=wavl,
                                     device_prefix=device_prefix,
                                     device_suffix=device_suffix,
                                     port_cross=port_cross,
                                     port_bar=port_bar,
                                     name=name,
                                     main_script_directory=results_directory)

            group_index.process_device_data()
            group_index.sort_devices_by_length()
            gindex, gindexError = group_index.plot_group_index(target_wavelength=wavl)
            group_index.plot_coupling_coefficient_contour()

            # Append the results to the list
            result_entry = {'Name': name, 'Wavelength': wavl, 'Polarization': pol, 'Data': gindex,
                            'Error': gindexError}
            # add error as val to return
            self.results_list.append(result_entry)

        # Convert the list of results to a DataFrame
        results_df = pd.DataFrame(self.results_list)

        return results_df

    def pdfReport(self, results_df):
        # Prompt user for chipname input
        chipname = input("Enter chip name: ")

        # Prompt user for measurement date input
        date_str = input("Enter measurement date (YYYY-MM-DD): ")

        # Validate and convert the date input
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            return

        # Prompt user for process input
        process = input("Enter process: ")

        # Create a PDF document
        doc = SimpleDocTemplate(f"{chipname}_analysis_report.pdf", pagesize=letter)

        # Create a story to add elements to the PDF
        story = []

        # Define the title
        title = f"Analysis Report of {chipname} chip"
        title_style = getSampleStyleSheet()["Title"]
        title_style.fontName = 'Times-Bold'
        title_text = Paragraph(title, title_style)
        story.append(title_text)

        # Add a paragraph of text
        paragraph = (f"<br/>Measurement date: {date} <br/><br/>"
                     f"Process: {process}<br/><br/>")  # text paragraph
        paragraph_style = getSampleStyleSheet()["Normal"]
        paragraph_style.fontName = 'Times-Roman'
        paragraph_style.fontSize = 12  # Change font size to 12
        paragraph_text = Paragraph(paragraph, paragraph_style)
        story.append(paragraph_text)

        # Create a table with the data from the DataFrame
        table_data = [results_df.columns.tolist()] + results_df.values.tolist()
        table = Table(table_data, colWidths=[1.5 * inch] * len(results_df.columns))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.ghostwhite),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Roman'),
            ('FONTSIZE', (0, 0), (-1, 0), 16),  # Title font size 16
            ('FONTSIZE', (0, 1), (-1, -1), 11),  # Table font size 12
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, 1), colors.whitesmoke),  # Background for header row
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        story.append(table)

        # Build the PDF
        doc.build(story)