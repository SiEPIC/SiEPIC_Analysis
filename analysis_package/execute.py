import os
import yaml
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime
from PyPDF2 import PdfWriter, PdfReader

from analysis_package import Device
from analysis_package.bragg import DirectionalCoupler
from analysis_package.groupIndex import GroupIndex
class Execute:
    def __init__(self, root_path):
        self.root_path = root_path
        self.results_list = []
        self.all_pdf_paths = []

    def analyze_cutback(self, dataset, results_directory):
        name = dataset['name']
        wavl = dataset['wavelength']
        pol = dataset['polarization']
        files_path = self.root_path
        target_prefix = dataset['target_prefix']
        target_suffix = dataset['target_suffix']
        characterization = dataset['characterization']
        port = dataset['port']

        # Create an instance of the Device class (Assuming you have a Device class)
        device = Device(wavl=wavl, pol=pol,
                        root_path=self.root_path,
                        main_script_directory=results_directory,
                        files_path=files_path,
                        target_prefix=target_prefix,
                        target_suffix=target_suffix,
                        port=port, name=name,
                        characterization=characterization)

        # Load data
        wavelengths_file, channel_pwr = device.loadData()

        # Process data
        lengths_cm, lengths_cm_sorted, lengths_um, input_to_function = device.process_data(wavelengths_file,
                                                                                           channel_pwr)
        separated_data = device.getSets(input_to_function, lengths_um)

        # Call the graphRaw method
        figure_data_raw, df_figures_raw = device.graphRaw(separated_data)

        # Extract power arrays and wavelength data
        power_arrays, wavelength_data = device.getArrays(input_to_function, lengths_um)

        # Calculate slopes
        slopes = device.getSlopes(power_arrays, lengths_cm_sorted, wavelength_data, target_wavelength=wavl)

        # Call the graphCutback method
        cutback_value, cutback_error, df_figures_cutback = device.graphCutback(wavl, wavelength_data, slopes)

        # Combine df_figures_raw and df_figures_cutback
        df_figures_combined = pd.concat([df_figures_raw, df_figures_cutback], ignore_index=True)

        # Initialize results list for this analysis
        self.results_list = []

        # Append the results to the list
        result_entry = {'Name': name, 'Wavelength': f'{wavl}nm', 'Polarization': pol, 'Data': cutback_value,
                        'Error': cutback_error, 'Characterization': characterization}
        self.results_list.append(result_entry)

        # Convert the list of results to a DataFrame
        results_df = pd.DataFrame(self.results_list)

        # Round the 'Cutback Value' and 'Error' columns to two decimal places in the DataFrame
        results_df['Data'] = results_df['Data'].round(2)
        results_df['Error'] = results_df['Error'].round(2)

        return results_df, df_figures_combined

    def analyze_bragg(self, dataset, results_directory):
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
             wavl=wavl,
             pol=pol,
             main_script_directory=results_directory
            )

        dc.process_files()
        dc.plot_devices()
        dc.plot_analysis_results()
        bragg_drift = dc.overlay_simulation_data(target_wavelength=wavl)

        # Initialize results list for this analysis
        self.results_list = []

        characterization = dataset['characterization']

        # Append the results to the list
        result_entry = {'Name': name, 'Wavelength': f'{wavl}nm', 'Polarization': pol, 'Data': bragg_drift,
                        'Error': 'N/A', 'Characterization': characterization}
        self.results_list.append(result_entry)

        # Convert the list of results to a DataFrame
        results_df = pd.DataFrame(self.results_list)

        # Round the 'Cutback Value' and 'Error' columns to two decimal places in the DataFrame
        results_df['Data'] = results_df['Data'].round(2)

        df_figures = dc.df_figures

        return results_df, df_figures

    def analyze_gIndex(self, dataset, results_directory):
        name = dataset['name']
        wavl = dataset['wavelength']
        pol = dataset['polarization']
        files_path = os.path.join(self.root_path, f"{wavl}_{pol}")
        device_prefix = dataset['device_prefix']
        device_suffix = dataset['device_suffix']
        port_cross = dataset['port_cross']
        port_bar = dataset['port_bar']

        group_index = GroupIndex(directory_path=files_path,
                                 wavl=wavl,
                                 pol=pol,
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

        # Initialize results list for this analysis
        self.results_list = []

        characterization = dataset['characterization']

        # Append the results to the list
        result_entry = {'Name': name, 'Wavelength': f'{wavl} nm', 'Polarization': pol, 'Data': gindex,
                        'Error': gindexError, 'Characterization': characterization}
        self.results_list.append(result_entry)

        # Convert the list of results to a DataFrame
        results_df = pd.DataFrame(self.results_list)

        # Round the 'Cutback Value' and 'Error' columns to two decimal places in the DataFrame
        results_df['Data'] = results_df['Data'].round(2)
        results_df['Error'] = results_df['Error'].round(2)

        df_figures = group_index.df_figures

        return results_df, df_figures

    def pdfReport(self, results_directory, results_df, df_figures):
        # Prompt user for chipname input
        chipname = input("Enter chip name: ")

        # Prompt user for measurement date input
        #date_str = input("Enter measurement date (YYYY-MM-DD): ")

        # Validate and convert the date input
        #try:
            #date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d, %Y")
        #except ValueError:
            #print("Invalid date format. Please use YYYY-MM-DD.")
            #return

        # Prompt user for process input
        # process = input("Enter process: ")

        # Create the full path for the PDF file including the results_directory
        pdf_path = os.path.join(results_directory, f"{chipname}_analysis_report.pdf")

        # Create a PDF document with a smaller top margin
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)  # You can adjust the second parameter for the desired top margin

        # Create a story to add elements to the PDF
        story = []

        # Define the title
        title = f"Analysis Report of {chipname} chip"
        title_style = getSampleStyleSheet()["Title"]
        title_style.fontName = 'Times-Bold'
        title_text = Paragraph(title, title_style)
        story.append(title_text)

        # Add a paragraph of text
        #paragraph = (f"<br/>Measurement date: {date} <br/><br/>"
                     #f"Process: {process}<br/><br/>")  # text paragraph
        paragraph = "text"
        paragraph_style = getSampleStyleSheet()["Normal"]
        paragraph_style.fontName = 'Times-Roman'
        paragraph_style.fontSize = 12  # Change font size to 12
        paragraph_text = Paragraph(paragraph, paragraph_style)
        story.append(paragraph_text)

        # Rename the columns in the DataFrame
        results_df = results_df.rename(columns={'Characterization': 'Analysis'})

        # Create a table with the data from the DataFrame
        table_data = [results_df.columns.tolist()] + results_df.values.tolist()

        # Specify different widths for each column (adjust these values as needed)
        col_widths = [1.5 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 2.0 * inch]

        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.ghostwhite),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Times-Roman'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),  # Title font size 14
            ('FONTSIZE', (0, 1), (-1, -1), 11),  # Table font size 12
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, 1), colors.whitesmoke),  # Background for header row
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        story.append(table)

        # Add a page break before adding the figure
        story.append(PageBreak())

        # Add the figures to the PDF, two figures per page
        figures_per_page = 2
        for i, row in df_figures.iterrows():
            # Access the figure and other information
            figure = row['Figure']

            # Convert BytesIO to Image
            img = Image(figure, width=7.2 * inch, height=4.32 * inch)

            # Add the image to the story
            story.append(img)

            # If the current figure is the last one on the page, add a page break
            if (i + 1) % figures_per_page == 0:
                story.append(PageBreak())

        # Build the PDF
        doc.build(story)

        # Return the path of the generated PDF
        return pdf_path

    def merge_pdfs(self, pdf_paths, output_path):
        pdf_writer = PdfWriter()

        for pdf_path in pdf_paths:
            print(f"Processing PDF: {pdf_path}")  # Debugging statement
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)

        with open(output_path, 'wb') as out_pdf:
            pdf_writer.write(out_pdf)

        print("PDFs combined successfully.")

    def gen_analysis(self):
        # Construct the path to your .yaml file using root_path
        yaml_file = os.path.join(self.root_path, 'config.yaml')

        # Load data from the .yaml file
        with open(yaml_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        # Create the results directory in the parent directory of root_path
        parent_directory = os.path.dirname(self.root_path)
        results_directory = os.path.join(parent_directory, "presuffix_results")

        # Check if the "analysis_results" folder exists, and create it if it doesn't
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)

        # Initialize an empty DataFrame to store results
        results_df = pd.DataFrame()

        # Initialize an empty DataFrame to store figures
        df_figures = pd.DataFrame()

        # Perform analysis based on characterization
        for dataset in data['devices']:
            characterization = dataset['characterization']

            if characterization in ['Insertion Loss (dB/cm)', 'Insertion Loss (dB/device)']:
                cutback_results_df, df_figures_cutback = self.analyze_cutback(dataset, results_directory)
                results_df = pd.concat([results_df, cutback_results_df], ignore_index=True)
                df_figures = pd.concat([df_figures, df_figures_cutback], ignore_index=True)
            elif characterization == 'Bragg Drift (nm)':
                bragg_results_df, df_figures_bragg = self.analyze_bragg(dataset, results_directory)
                results_df = pd.concat([results_df, bragg_results_df], ignore_index=True)
                df_figures = pd.concat([df_figures, df_figures_bragg], ignore_index=True)
            elif characterization == 'Group Index':
                gindex_results_df, df_figures_gindex = self.analyze_gIndex(dataset, results_directory)
                results_df = pd.concat([results_df, gindex_results_df], ignore_index=True)
                df_figures = pd.concat([df_figures, df_figures_gindex], ignore_index=True)
            else:
                print(f"Unknown characterization type: {characterization}")

        # Generate PDF report and collect its path
        report_path = self.pdfReport(results_directory, results_df, df_figures)

        return results_df, report_path