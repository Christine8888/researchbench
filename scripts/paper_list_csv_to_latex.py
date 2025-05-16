import csv
import re
import sys
import argparse

def sanitize_for_latex(text):
    """
    Properly sanitize text for LaTeX by handling special characters in a way
    that won't cause compilation errors.
    """
    # First handle backslashes and other special characters
    replacements = {
        '\\': '\\textbackslash{}',
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}'
    }
    
    # Apply replacements carefully to avoid cascading replacements
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text

def csv_to_latex(csv_file, output_file=None, add_citations=True):
    # Read CSV data
    papers_data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            papers_data = list(reader)
            
            # Sort the papers data case-insensitively by paper_id
            papers_data.sort(key=lambda x: x['paper_id'].lower())
            
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Start building LaTeX table
    latex_content = []
    
    # Table header
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Summary of papers and their tasks.}")
    latex_content.append("\\label{tab:paper_analysis}")
    
    # Table with narrower columns
    latex_content.append("\\begin{tabular}{lp{7.5cm}cc}")
    latex_content.append("\\toprule")
    latex_content.append("\\textbf{Paper ID} & \\textbf{Title} & \\textbf{\\# Tasks} & \\textbf{Avg. Diff.} \\\\")
    latex_content.append("\\midrule")
    
    # Table rows
    for paper in papers_data:
        # Format paper_id as code text with smaller font
        raw_paper_id = paper['paper_id']
        paper_id = f"\\small\\texttt{{{sanitize_for_latex(raw_paper_id)}}}"
        
        # Italicize the title with smaller font
        title = f"\\small\\textit{{{sanitize_for_latex(paper['title'])}}}"
        
        task_count = paper['task_count']
        avg_difficulty = paper['avg_difficulty']
        
        # Add citation command to the title if requested
        if add_citations:
            # Keep original paper_id with underscores for citation key
            citation_key = raw_paper_id
            title = f"{title} \\citep{{{citation_key}}}"
        
        # Do not truncate titles - the p{7.5cm} column will handle wrapping
        latex_content.append(f"{paper_id} & {title} & {task_count} & {avg_difficulty} \\\\")
    
    # Table footer
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Join all lines
    full_latex = "\n".join(latex_content)
    
    # Write to file if output file is specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_latex)
        print(f"LaTeX table written to {output_file}")
    
    # Always print to console
    print("Here is the LaTeX code for copy-pasting:")
    print("----------------------------------------")
    print(full_latex)
    print("----------------------------------------")

if __name__ == "__main__":
    """
    # With citations (default)
    python utils/paper_list_csv_to_latex.py paper_list.csv -o papers/papers_table.tex

    # Without citations
    python utils/paper_list_csv_to_latex.py paper_list.csv -o papers/papers_table.tex --no-citations
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert CSV paper analysis to LaTeX table")
    parser.add_argument("csv_file", help="Path to the CSV file to convert")
    parser.add_argument("-o", "--output", help="Output LaTeX file path (optional)")
    parser.add_argument("--no-citations", action="store_true", help="Disable adding citation commands")
    
    args = parser.parse_args()
    
    # Call function with provided arguments
    csv_to_latex(args.csv_file, args.output, not args.no_citations)