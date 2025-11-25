A collection of data analysis tools for silicon photonics measurements.

1) openEBL_viewer:
  - GUI to load layout and measurement files, and display individual devices and plot the measurements

2) Test structure compact model extraction
   
Takes in .csv files from test stations and performs:
- cutback calculations
- bragg drift calculations
- group index calculations

Devices analyzed:
- Straight waveguides
- Spiral waveguides
- SWG
- SWG assist
- Y-branch
- DC
- bragg(nm)
- bragg(dW)


**main.py is inside SampleData

Install package before use through the following command:
pip install -e .


To be merged with PWB_calibration
