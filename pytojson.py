import os
import json

def create_json_from_python_files(folder_path, output_json_filename="code_dump.json"):
    """
    Creates a JSON file where keys are Python file names and values are the code content.

    Args:
        folder_path (str): The path to the folder containing the Python files.
        output_json_filename (str): The name of the output JSON file.
    """
    code_dictionary = {}
    
    # 1. Iterate through the files in the folder
    print(f"Scanning folder: {folder_path}")
    for filename in os.listdir(folder_path):
        # 2. Check if the file is a Python file
        if filename.endswith(".py"):
            file_path = os.path.join(folder_path, filename)
            
            # Skip directories that might match the pattern (though unlikely with os.listdir)
            if os.path.isfile(file_path):
                try:
                    # 3. Read the content of the Python file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                        
                    # 4. Add the filename and code content to the dictionary
                    code_dictionary[filename] = code_content
                    print(f"  ✅ Added {filename}")
                    
                except Exception as e:
                    print(f"  ❌ Could not read file {filename}: {e}")
                    
    # 5. Write the dictionary to a JSON file
    print(f"\nWriting to {output_json_filename}...")
    try:
        with open(output_json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(code_dictionary, json_file, indent=4)
        print(f"🎉 Successfully created {output_json_filename} with {len(code_dictionary)} files.")
    except Exception as e:
        print(f"🚨 Error writing JSON file: {e}")

# --- Example Usage ---
# IMPORTANT: Replace 'path/to/your/folder' with the actual path!
# For example, if your folder is in the same directory as this script:
# FOLDER_TO_SCAN = "." 
# Or an absolute path:
# FOLDER_TO_SCAN = "/Users/username/my_project/src"

FOLDER_TO_SCAN = "beardstylegan" 
OUTPUT_FILE = "python_code_map.json"

create_json_from_python_files(FOLDER_TO_SCAN, OUTPUT_FILE)