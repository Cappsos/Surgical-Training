import os

def clean_and_modify_polygons(polygon_folder, output_folder):
    """
    Clean and modify polygon .txt files in a folder:
    1. Keep all lines in each file.
    2. Add a '0' label at the beginning of each line.
    3. Keep empty files intact.
    4. Save the modified files as copies in a new folder.

    Parameters:
        polygon_folder: Path to the folder containing the original polygon .txt files.
        output_folder: Path to the folder where modified copies will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(polygon_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(polygon_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            # Read the contents of the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify lines by adding '0 ' at the beginning of each line
            modified_lines = [f"0 {line.strip()}\n" for line in lines]

            # Ensure empty files remain intact
            if not modified_lines:
                print(f"Empty file: {file_name} remains unchanged.")
                open(output_file_path, 'w').close()  # Create an empty file in output folder
                continue

            # Write the modified content to the new file
            with open(output_file_path, 'w') as file:
                file.writelines(modified_lines)

            print(f"Processed: {file_name} -> {output_file_path}")

if __name__ == "__main__":
    # Example usage
    polygon_folder = "/root/surgical_training/output_segmented_pinze"  # Replace with your polygon folder path
    output_folder = "/root/surgical_training/output_segmented_pinze_modified"  # Folder for modified copies
    clean_and_modify_polygons(polygon_folder, output_folder)