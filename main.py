from analysis_package import Execute

# Example usage:
# include yaml file
if __name__ == "__main__":
    # Set the root_path and create the "analysis_results" folder
    # root_path = "D:\Academics\PyCharmProjects\Data\Actives-May-2023"
    root_path = "D:\Academics\PyCharmProjects\Data\Actives-May-2023"
    # root_path = "D:\Academics\PyCharmProjects\Data\groupindex_ex"
    analyzer = Execute(root_path)
    results_df = analyzer.analyze_cutback()
    print(results_df)
    # analyzer.analyze_bragg()
    # analyzer.analyze_gIndex()
