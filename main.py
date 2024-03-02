import os
from analysis_package import Execute

# Example usage:
if __name__ == "__main__":
    all_reports = []

    root_path = r"D:\Academics\PyCharmProjects\Data\Actives-May-2023\1500_TE"
    # root_path = "D:\Academics\PyCharmProjects\Data\groupindex_ex"
    # root_path = "D:\Academics\PyCharmProjects\Data\ZEP3chip4"
    # root_path = "D:\Academics\PyCharmProjects\Data\CMC chip\Run2"
    analyzer_1 = Execute(root_path)
    # results_df = analyzer.analyze_cutback()
    results_df_1, report_path = analyzer_1.gen_analysis()
    print(results_df_1)
    all_reports.append(report_path)
    # analyzer.analyze_bragg()
    # analyzer.analyze_gIndex()

    root_path = r"D:\Academics\PyCharmProjects\Data\Actives-May-2023\1310_TE"
    analyzer_2 = Execute(root_path)
    results_df_2, report_path = analyzer_2.gen_analysis()
    print(results_df_2)
    all_reports.append(report_path)

    # Combine the PDFs after all runs if needed
    combined_pdf_path = os.path.join(os.path.dirname(root_path), "RUNoFanalysis_results", "combined.pdf")
    analyzer_1.merge_pdfs(all_reports, combined_pdf_path)
    # changes