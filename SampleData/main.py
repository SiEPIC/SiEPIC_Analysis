import os
from analysis_package import Execute

# Example usage:
if __name__ == "__main__":
    all_reports = []

    chip_name = 'EdX_May_2023_1550TE'
    measure_date = '2023-08-28'  # YYYY-MM-DD
    process = 'ANT'

    # main not in project
    root_path = os.path.join(os.getcwd(), '1550_TE')
    analyzer_1 = Execute(root_path, chip_name, measure_date, process)
    results_df_1, report_path = analyzer_1.gen_analysis()
    print(results_df_1)
    all_reports.append(report_path)

    chip_name = 'EdX_May_2023_1310TE'
    root_path = os.path.join(os.getcwd(), '1310_TE')
    analyzer_2 = Execute(root_path, chip_name, measure_date, process)
    results_df_2, report_path = analyzer_2.gen_analysis()
    print(results_df_2)
    all_reports.append(report_path)

    chip_name = 'EdX_May_2023_1550TM'
    root_path = os.path.join(os.getcwd(), '1550_TM')
    analyzer_3 = Execute(root_path, chip_name, measure_date, process)
    results_df_3, report_path = analyzer_3.gen_analysis()
    print(results_df_3)
    all_reports.append(report_path)

    chip_name = 'EdX_May_2023_1310TM'
    root_path = os.path.join(os.getcwd(), '1310_TM')
    analyzer_4 = Execute(root_path, chip_name, measure_date, process)
    results_df_4, report_path = analyzer_4.gen_analysis()
    print(results_df_4)
    all_reports.append(report_path)

    # Combine the PDFs after all runs if needed
    combined_pdf_path = os.path.join(os.path.dirname(root_path), "analysis_results", "combined_results.pdf")
    analyzer_1.merge_pdfs(all_reports, combined_pdf_path)
