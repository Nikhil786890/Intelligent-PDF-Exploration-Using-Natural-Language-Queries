import pandas as pd
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Path to your cleaned CSV file
csv_file = 'imdb_top_1000_cleaned.csv'

# Read the CSV file into a DataFrame
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
    exit()

# Prepare the data for ReportLab Table
data = [list(df.columns)] + df.values.tolist()

# Create a PDF document
doc = SimpleDocTemplate("imdb_top_1000.pdf", pagesize=landscape(A4))
styles = getSampleStyleSheet()

# Define the style for the table
style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('BOX', (0, 0), (-1, -1), 1, colors.black),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 5), # Smaller font size to fit more content
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
])

# Define column widths as a percentage of the total table width. The numbers must add up to 1.0.
col_widths = [0.03, 0.15, 0.04, 0.04, 0.04, 0.05, 0.04, 0.15, 0.04, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

# Create the table with percentage widths
table = Table(data, colWidths=[x * landscape(A4)[0] for x in col_widths])
table.setStyle(style)

# Add the table to the document
elements = [table]
doc.build(elements)

print("PDF file created successfully with ReportLab. All columns should now fit on the page.")