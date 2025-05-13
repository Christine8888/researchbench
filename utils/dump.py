#!/usr/bin/env python3
import json
import sys
import argparse

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Add text to a JSON file field')
    parser.add_argument('file', help='JSON file to modify')
    parser.add_argument('field', help='Field name to update')
    parser.add_argument('--output', '-o', help='Output file (default: overwrite input file)', default=None)
    parser.add_argument('--input', '-i', help='Input text file (default: read from stdin)', default=None)
    
    args = parser.parse_args()
    
    # Load the JSON file
    try:
        with open(args.file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{args.file}' is not a valid JSON file")
        sys.exit(1)
    
    # Get the text content
    if args.input:
        try:
            with open(args.input, 'r') as f:
                text_content = f.read()
        except FileNotFoundError:
            print(f"Error: Input file '{args.input}' not found")
            sys.exit(1)
    else:
        print("Enter or paste your text (Ctrl+D or Ctrl+Z on a new line to finish):")
        text_content = sys.stdin.read()
    
    # Update the JSON data
    field_parts = args.field.split('.')
    target = data
    
    # Navigate to nested fields if needed
    for part in field_parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    
    # Set the value
    target[field_parts[-1]] = text_content
    
    # Save the output
    output_file = args.output if args.output else args.file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated field '{args.field}' in '{output_file}'")

if __name__ == '__main__':
    main()