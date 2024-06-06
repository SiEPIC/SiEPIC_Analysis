import os
from analysis_package import Execute

# Example usage:
if __name__ == "__main__":
    all_reports = []

    root_path = r"D:\Documents\Uni\Year_4COOP\Docs\Analysis_Scripts\package\SampleData\1550_TE"
    analyzer_1 = Execute(root_path)
    results_df_1, report_path = analyzer_1.gen_analysis()
    print(results_df_1)
    all_reports.append(report_path)

    """
    # main not in project
    root_path = os.path.join(os.getcwd(),'1550_TE')
    analyzer_1 = Execute(root_path)
    results_df_1, report_path = analyzer_1.gen_analysis()
    print(results_df_1)
    all_reports.append(report_path)

    root_path = os.path.join(os.getcwd(),'1550_TM')
    analyzer_2 = Execute(root_path)
    results_df_2, report_path = analyzer_2.gen_analysis()
    print(results_df_2)
    all_reports.append(report_path)

    # Combine the PDFs after all runs if needed
    combined_pdf_path = os.path.join(os.path.dirname(root_path), "analysis_results", "combined_results.pdf")
    analyzer_1.merge_pdfs(all_reports, combined_pdf_path)
    """