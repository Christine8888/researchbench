#!/usr/bin/env python
import re
import sys
import os

def clean_paper(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Keep track of section labels
    section_labels = {}
    label_pattern = r'\\label\{(sec:[^}]+)\}'
    for match in re.finditer(label_pattern, text):
        label = match.group(1)
        section_labels[label] = True
    
    # Check if this is LaTeX format
    is_latex = '\\begin{document}' in text or '\\documentclass' in text
    
    if is_latex:
        # Preserve abstracts before cleaning
        abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, flags=re.DOTALL)
        abstract_text = abstract_match.group(1).strip() if abstract_match else ""
        
        # Remove LaTeX preamble (everything before \begin{document})
        document_match = re.search(r'\\begin\{document\}', text)
        if document_match:
            text = text[document_match.start():]
        
        # Remove title, authors, and affiliations (more variations)
        # text = re.sub(r'\\title(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\author(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\address(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\affiliation(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\altaffiliation(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\email(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\orcid(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\correspondingauthor(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\institute(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\maketitle', '', text)
        text = re.sub(r'\\usepackage(\[.*?\])?\{.*?\}', '', text, flags=re.DOTALL)
        
        # Remove abstract environment but keep abstract text if we want to
        if abstract_text and False:  # Set to True if you want to keep the abstract
            # Be careful with backslashes in the replacement string
            replacement = r'\\begin{abstract}' + '\n' + abstract_text + '\n' + r'\\end{abstract}'
            text = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', replacement, text, flags=re.DOTALL)
        
        # Remove bibliography/references (handles multiple formats)
        text = re.sub(r'\\bibliographystyle\{.*?\}', '', text)
        text = re.sub(r'\\bibliography\{.*?\}', '', text)
        text = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\begin\{references\}.*?\\end\{references\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\section\*?\{References\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\section\*?\{REFERENCES\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\section\*?\{Bibliography\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        
        # Remove citation commands but preserve sentence structure
        # text = re.sub(r'(?:\\cite|\\citep|\\citet)(?:\[.*?\])?\{[^}]*\}', '', text)
        
        # Remove acknowledgments (multiple variations)
        text = re.sub(r'\\section\*?\{Acknowledge?ments?\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\subsection\*?\{Acknowledge?ments?\}.*?(?=\\section|\\subsection|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\paragraph\{Acknowledge?ments?\}.*?(?=\\paragraph|\\section|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\begin\{acknowledgements?\}.*?\\end\{acknowledgements?\}', '', text, flags=re.DOTALL)
        
        # Remove software sections
        text = re.sub(r'\\software\{.*?\}', '', text, flags=re.DOTALL)
        
        # Remove data availability and similar sections
        text = re.sub(r'\\section\*?\{Data [Aa]vailability\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\section\*?\{Code [Aa]vailability\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\section\*?\{Availability of [Dd]ata\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\section\*?\{[Dd]ata [Aa]ccess\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        
        # Remove comments (both inline and full-line)
        text = re.sub(r'(?m)^%.*$', '', text)  # Full line comments
        text = re.sub(r'(?<!\\)%.*?(?=\n)', '', text, flags=re.MULTILINE)  # Inline comments
        
        # Remove funding information
        text = re.sub(r'\\section\*?\{Funding\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\section\*?\{Financial [Ss]upport\}.*?(?=\\section|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\\funding\{.*?\}', '', text, flags=re.DOTALL)
        
        # Remove footnotes
        text = re.sub(r'\\footnote\{.*?\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\thanks\{.*?\}', '', text, flags=re.DOTALL)
        
        # Remove bibtex entries
        text = re.sub(r'@[a-zA-Z]+\{[^,]+,.*?(?=@|\Z)', '', text, flags=re.DOTALL)
        
    else:
        # For non-LaTeX papers, try to remove standard sections
        text = re.sub(r'References\s*\n-+\n.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)

     # Clean up multiple blank lines, extra spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Make sure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Cleaned paper saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_paper.py input_file output_file")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        clean_paper(input_path, output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        sys.exit(1)