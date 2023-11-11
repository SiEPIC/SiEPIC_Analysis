from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch  # Import the 'inch' unit

def pdfReport(chip):
    # Create a PDF document
    doc = SimpleDocTemplate(f"{chip}_analysis_report.pdf", pagesize=letter)

    # Create a story to add elements to the PDF
    story = []

    # Define the title
    title = f"Analysis Report of {chip} chip"
    title_style = getSampleStyleSheet()["Title"]
    title_style.fontName = 'Times-Bold'
    title_text = Paragraph(title, title_style)
    story.append(title_text)

    # Add a paragraph of text
    paragraph = "This is a sample report with a table:<br/><br/>"  # text paragraph
    paragraph_style = getSampleStyleSheet()["Normal"]
    paragraph_style.fontName = 'Times-Roman'
    paragraph_style.fontSize = 12  # Change font size to 12
    paragraph_text = Paragraph(paragraph, paragraph_style)
    story.append(paragraph_text)

    # Define table data
    table_data = [
        ["Name", f"{chip}"],  # row1
        ["Measurements:", "spiral1550", "straight1550", "spiral1310", "straight1310"],  # row2
        ["Analysis", "", "", "", ""],  # row3
        ["", "", "", "", ""],  # row4
        ["", "", "", "", ""],  # row5
    ]

    # Create a table with the data
    table = Table(table_data, colWidths=[1.5 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.ghostwhite),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Roman'),
        ('FONTSIZE', (0, 0), (-1, 0), 16),  # Title font size 16
        ('FONTSIZE', (0, 1), (-1, -1), 11),  # Table font size 12
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('SPAN', (1, 0), (4, 0)),  # Span the cell after "Name"
    ]))

    story.append(table)

    # Build the PDF
    doc.build(story)

# Example usage:
chip = "example"
pdfReport(chip)
