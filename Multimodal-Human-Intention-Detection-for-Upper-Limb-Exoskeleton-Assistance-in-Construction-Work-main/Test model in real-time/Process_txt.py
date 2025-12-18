import os
import re

def process_file(file_path):
    """Read a .txt file, apply offset and renumbering logic, then save a modified version."""

    # Read file content
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    offset_active = False
    offset_value = 0.0
    sample_index = 1

    for line in lines:
        match = re.match(r"(Sample_\d+)\s*:\s*(.+?)\s+at\s+([\d.]+)(?:\s*\(-\s*([\d.]+)\))?", line.strip())
        if match:
            sample_name, description, time_value, offset = match.groups()
            time_value = float(time_value)

            # When an offset is detected → activate offset mode and reset numbering
            if offset:
                offset_value = abs(float(offset))
                offset_active = True
                sample_index = 1
                time_value -= offset_value
                new_line = f"Sample_{sample_index} : {description} at {time_value:.2f}\n"
                sample_index += 1

            # Stop offset mode if Sample_1 is encountered
            elif sample_name.strip() == "Sample_1":
                offset_active = False
                new_line = f"{line.strip()}\n"

            # Continue offset mode (shifted times, new numbering)
            elif offset_active:
                time_value -= offset_value
                new_line = f"Sample_{sample_index} : {description} at {time_value:.2f}\n"
                sample_index += 1

            # Normal line (no offset active)
            else:
                new_line = f"{line.strip()}\n"

            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Save modified file next to the original
    folder = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    new_file_path = os.path.join(folder, f"{name}_modified{ext}")

    with open(new_file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"✅ Modified file saved as: {new_file_path}")


if __name__ == "__main__":
    base_folder = os.path.dirname(os.path.abspath(__file__))  # "Test model in real-time"

    folder_name = input("Enter the name of the folder containing the .txt file: ").strip()
    folder_path = os.path.join(base_folder, folder_name)

    # Check folder existence
    if not os.path.isdir(folder_path):
        print("❌ Error: Folder not found. Please check the name and try again.")
        exit()

    file_name = input("Enter the name of the .txt file (with or without extension): ").strip()
    if not file_name.lower().endswith(".txt"):
        file_name += ".txt"
    file_path = os.path.join(folder_path, file_name)

    # Check file existence
    if not os.path.isfile(file_path):
        print("❌ Error: File not found in this folder. Please verify the file name.")
        exit()

    # Process file if everything is valid
    process_file(file_path)


