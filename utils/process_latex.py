import re
import json
import os

def process_latex(text: str, split: str = 'partial') -> str:
    """
    Process LaTeX source with regex to remove figures, non-text content, comments, etc.
    
    Parameters:
        text (str): The LaTeX source text
        split (str): One of 'all', 'partial', 'minimal'
            - 'minimal': keep just the abstract
            - 'partial': keep everything from abstract to references, removing figures and appendices
            - 'all': keep everything in the paper except comments and formatting
    
    Returns:
        str: The processed LaTeX text
    """
    # remove lines that are comments
    text = re.sub(r"^[\s]*%.*$", "", text, flags=re.MULTILINE)
    
    # remove inline comments (careful to preserve the rest of the line)
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    # only remove figures if split is not 'all'
    if split != 'all':
        # remove \begin{figure}...\end{figure} or \begin{subfigure}...\end{subfigure} or \begin{figure*}...\end{figure*}
        text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", text, flags=re.DOTALL)
        text = re.sub(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", "", text, flags=re.DOTALL)
        text = re.sub(r"\\begin\{figure\*\}.*?\\end\{figure\*\}", "", text, flags=re.DOTALL)

    # remove extra newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # remove empty lines with only whitespace
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)

    if "\\begin{document}" in text and "\\end{document}" in text:
        # capturing group
        parts = re.split(r"(\\begin\{document\}.*?\\end\{document\})", text, flags=re.DOTALL)
        text = parts[1]

    # extract different sections based on split parameter
    if split == 'minimal':
        # keep just abstract
        abstract_pattern = r"\\begin\{abstract\}(.*?)\\end\{abstract\}"
        abstract_match = re.search(abstract_pattern, text, flags=re.DOTALL)
        if abstract_match:
            return abstract_match.group(0)
        else:
            # try alternate abstract format
            abstract_pattern = r"\\abstract(.*?)(?:\\section|\\end)"
            abstract_match = re.search(abstract_pattern, text, flags=re.DOTALL)
            if abstract_match:
                return "\\abstract" + abstract_match.group(1)
            return ""  # Return empty string if no abstract found
    
    elif split == 'partial':
        # keep everything from abstract to references/acknowledgements
        abstract_match = re.search(r"\\begin\{abstract\}|\\abstract", text, flags=re.IGNORECASE)
        
        # look for end markers
        ack_match = re.search(r"\\section\*?\{acknowledgements?\}|\\acknowledgements", text, flags=re.IGNORECASE)
        ref_match = re.search(r"\\bibliography\{.*?\}|\\begin\{thebibliography\}|\\section\*?\{references\}", text, flags=re.IGNORECASE)
        
        # determine end position
        end_pos = len(text)
        if ack_match:
            end_pos = ack_match.start()
        elif ref_match:
            end_pos = ref_match.start()
            
        if abstract_match:
            start_pos = abstract_match.start()
            return text[start_pos:end_pos]
        return text  # Return full text if no abstract found
    
    # for 'all', return the entire text with comments removed
    return text

def process_file(file_path: str, out_txt: str = None, out_json: str = None, field: str = 'instructions', split: str = 'partial') -> str:
    """
    Process a txt file with LaTeX source, output into existing json with cleaned instructions
    
    Parameters:
        file_path (str): Path to the LaTeX file
        out_txt (str, optional): Path to output text file
        out_json (str, optional): Path to output JSON file
        field (str): The field to save in the JSON file
        split (str): One of 'all', 'partial', 'minimal'
    
    Returns:
        str: The processed LaTeX text
    """
    if split not in ['all', 'partial', 'minimal']:
        raise ValueError(f"Invalid split option: {split}. Must be one of ['all', 'partial', 'minimal']")
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    clean_text = process_latex(text, split)
    
    if out_json:
        # save to json
        with open(out_json, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {out_json}")
                existing_data = {}
            
            existing_data[field] = clean_text
            
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)
    
    if out_txt:
        # save to txt
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(clean_text)
    
    return clean_text

if __name__ == "__main__":
    import argparse
    # command line utility for processing LaTeX files to instruction text
    parser = argparse.ArgumentParser(description="process LaTeX files to instruction text")
    parser.add_argument("input_file", help="input TeX source path")
    parser.add_argument("--output", "-o", help="output JSON file path")
    parser.add_argument("--split", "-s" , choices=['all', 'partial', 'minimal'], 
                        default='all', help="text chunking option")
    parser.add_argument("--output_txt", "-t", help="output text file path")
    args = parser.parse_args()
    
    result = process_file(file_path=args.input_file, out_json=args.output, out_txt=args.output_txt, split=args.split)
    
    if not args.output and not args.output_txt:
        print(result)