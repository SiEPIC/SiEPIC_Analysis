from analysis_package import Execute

# Example usage:
if __name__ == "__main__":
    root_path = "D:\\Academics\\PyCharmProjects\\Data\\Actives-May-2023"
    analyzer = Execute(root_path)
    analyzer.load_and_analyze()
