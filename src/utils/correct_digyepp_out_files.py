import os
import json

def filter_json_lines(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Check if the current item is a file
        if os.path.isfile(input_path):
            with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    try:
                        # Parse the line as a JSON object
                        json_obj = json.loads(line)
                        # Check if the "predicted_ner" attribute is present
                        if 'predicted_ner' in json_obj:
                            # Write the valid line to the output file
                            outfile.write(line)
                    except json.JSONDecodeError:
                        # Skip lines that aren't valid JSON
                        print(f"Invalid JSON in file {filename}: {line.strip()}")
                        continue

# Specify the input and output folder paths
input_folder = 'D:\GitRepos\GitRepos\SKG-pipeline\outputs\dygiepp_output'
output_folder = 'D:\GitRepos\GitRepos\SKG-pipeline\outputs\dygiepp_output_corrected'


if __name__ == '__main__':
    # Run the function
    filter_json_lines(input_folder, output_folder)
