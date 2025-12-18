# TCN_Local_Prediction_Final.py

if __name__ == "__main__":
    print("\033cStarting ...\n")  # Clear terminal

# ----   Modifiable variables   ----
action_to_idx = {
    'Bimanual_Down': 0,
    'Bimanual_Left': 1,
    'Bimanual_Prepare': 2,
    'Bimanual_Right': 3,
    'Bimanual_Up': 4,
    'Unimanual_Down': 5,
    'Unimanual_Left': 6,
    'Unimanual_Prepare': 7,
    'Unimanual_Right': 8,
    'Unimanual_Up': 9
}
root_directory = 'Temporary_Data'  # Directory for incoming samples
time_for_prediction = 25
prediction_threshold = 3

# ------------------------------------
import os
import sys
import time
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    import torch._dynamo
except ModuleNotFoundError as Err:
    missing_module = str(Err).replace('No module named ', '').replace("'", "")
    sys.exit(f'No module named {missing_module} try: pip install {missing_module}')

try:
    from Imports.dataloader_3 import HARDataSet
    from Imports.Models.fusion_tcn_3 import FusionModelTCN
    from Imports.Models.MoViNet.config import _C as config
except Exception as e:
    sys.exit(f"Import Error: {e}")

try:
    from Imports.Safe_Functions import model_exist, detect_tools_with_fusion_check
except ModuleNotFoundError:
    # fallback: define simple placeholders
    def model_exist(): return True
    def detect_tools_with_fusion_check(yolo, frames, predicted_label): return ("None", False)

torch._dynamo.config.suppress_errors = True

# Ask if we should record predictions
record_choice = input("Recording data? (Y/N): ").strip().upper()
log_file = None
if record_choice == "Y":
    txt_name = input("Enter txt filename (without extension) in TCN folder: ").strip()
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Test model in real-time", "TCN")
    os.makedirs(base_dir, exist_ok=True)
    log_file = os.path.join(base_dir, txt_name + ".txt")
    print(f"Predictions will be recorded in: {log_file}")

def make_prediction(Dataset):
    Loader = DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    with torch.no_grad():
        for video_frames, imu_data in Loader:
            video_frames, imu_data = video_frames.to(device), imu_data.to(device)
        
            out = model(video_frames, imu_data)  # (B, num_classes)
            predicted = torch.argmax(out, dim=1)
            predicted_label = idx_to_action[predicted.item()]

            from Imports.Safe_Functions import detect_tools_with_fusion_check
            final_tool, needs_support = detect_tools_with_fusion_check(yolo_model, video_frames, predicted_label)

    return predicted, final_tool, needs_support

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

# If there is no model to load, we stop
if not model_exist() :
    sys.exit("No model to load") # If there is no model to load, we stop

try:
    Done = False
    print("Waiting for data, launch data collection file")  # Printed once
    while not Done:
        try:
            if len(os.listdir(root_directory)) > 2:
                Done = True
            else:
                time.sleep(0.1)
        except FileNotFoundError:
            pass
        time.sleep(0.1)
except KeyboardInterrupt:
    sys.exit('\nProgramme Stopped\n')

'''input('Programme Ready, Press Enter to Start')
for i in range(3) :
    print(f'Starting in {3-i}s')
    time.sleep(1)
    print(LINE_UP, end=LINE_CLEAR)'''

Start_Tracking_Time = time.time()

idx_to_action = {v: k for k, v in action_to_idx.items()}
tracking = []

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

dataset = HARDataSet(root_dir=root_directory, transform=transform, sequence_length=10)

# Fusion model path and parameters
ModelToLoad_Path = r"D:\Bristol Robotics Laboratory\AI Project\Multimodal-Human-Intention-Detection-for-Upper-Limb-Exoskeleton-Assistance-in-Construction-Work-main\Pre Trained Model\fusion_movinet_tcn_final_9.pt"   #9
ModelName = os.path.basename(ModelToLoad_Path).replace('.pt', '')

# Yolo model path
yolo_model_path = r"D:\Bristol Robotics Laboratory\AI Project\Multimodal-Human-Intention-Detection-for-Upper-Limb-Exoskeleton-Assistance-in-Construction-Work-main\Pre Trained Model\best.pt"
yolo_model_name = os.path.basename(yolo_model_path).replace('.pt', '')

if ModelName.endswith('.pt') :
    ModelName = ModelName.replace('.pt','')
else :
    ModelName = ModelName.replace('.pht','')
print(f"Loading Fusion model: {ModelName}")
print(f"Loading YOLO model: {yolo_model_name}")
yolo_model = YOLO(yolo_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


# load model
model = FusionModelTCN(movinet_config=config.MODEL.MoViNetA0,
                       num_classes=len(action_to_idx),
                       tcn_input_size=27,
                       tcn_hidden_size=512,
                       tcn_num_layers=6,
                       tcn_dropout=0.4,
                       proj_imu_size=512,
                       freeze_backbone=False)
model.load_state_dict(torch.load(ModelToLoad_Path, weights_only = True, map_location=device))
model.to(device)
model.eval()

processed_samples = set() # To keep track of processed samples
last_output = {"label": None, "tool": None} # Store last output to avoid repetition
last_action = None

try: #Main Loop
    print(f'\033cProgramme running   ctrl + C to stop\n\nLoading {ModelName}\nUsing {device}\n')
    old_sample = ''
    first_sample = ''
    for action in action_to_idx:
        tracking.append(0)

    if not os.listdir(root_directory):
        print('No files in root directory')
        sys.exit(0)

    while True:
        dataset = HARDataSet(root_dir=root_directory, transform=transform, sequence_length=10)
        if len(dataset.samples) < 2:
            time.sleep(1)
            continue

        # Sorted list of files
        sample_folders = sorted(
            list({os.path.basename(os.path.dirname(s[0][0])) for s in dataset.samples}),
            key=lambda x: int(x.replace("Sample_", ""))
        )

        # We ignore the last sample (still filling up)
        samples_to_process = sample_folders[:-1]

        for sample_folder in samples_to_process:
            if sample_folder in processed_samples:
                continue
            
            sample_number = int(sample_folder.replace("Sample_", ""))

            sample_indices = [i for i, s in enumerate(dataset.samples)
                              if os.path.basename(os.path.dirname(s[0][0])) == sample_folder]

            if not sample_indices:
                continue

            idx = sample_indices[0]
            try:
                frames_tensor, imu_tensor = dataset[idx]
                with torch.no_grad():
                    frames_tensor, imu_tensor = frames_tensor.unsqueeze(0).to(device), imu_tensor.unsqueeze(0).to(device)
                    prediction = torch.argmax(model(frames_tensor, imu_tensor))
                    label = idx_to_action.get(prediction.item(), "Rest")

                    from Imports.Safe_Functions import detect_tools_with_fusion_check
                    final_tool, needs_support = detect_tools_with_fusion_check(yolo_model, frames_tensor, label)

                tracking[prediction] += 1

                #Display all actions

                # --- Display predictions depending on recording mode ---
                if record_choice == "Y":
                    # Always display all samples
            #        if needs_support:
            #           print("Detected bimanual action -> Providing long roller equivalent torque support")

                    print(f'{sample_folder} : {label} | Tool: {final_tool} at {round(time.time()-Start_Tracking_Time,2)}')
                    last_output["label"] = label  # update label each time
                else:
                    # Display only if the prediction is different from the last one
                    if label != last_output["label"]:
               #         if needs_support:
                #            print("Detected bimanual action -> Providing long roller equivalent torque support")

                        print(f'{sample_folder} : {label} | Tool: {final_tool} at {round(time.time()-Start_Tracking_Time,2)}')
                        last_output["label"] = label  # update only if label changes
                
                # When recording data : --- TXT logging: only first 5 samples --- (For Unimanual actions)

                if log_file and sample_number <= 5:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"{sample_folder} : {label} | Tool: {final_tool} at {round(time.time()-Start_Tracking_Time,2)}\n")
                    # Stop logging after Sample_5
                    if sample_number == 5:
                        print("✅ TXT logging finished after Sample_5")
                        log_file = None

                if first_sample == '':
                    first_sample = sample_folder

                processed_samples.add(sample_folder)

            except Exception as e:
                print(f'Error on {sample_folder}')
                traceback.print_exc()
                continue

except KeyboardInterrupt:
    pass

except FileNotFoundError:
    print("Samples folder got deleted")
        
num_of_predictions = sum(tracking)
num_first = int(first_sample.replace('Sample_',''))

if processed_samples:
    last_sample = sorted(processed_samples, key=lambda x: int(x.replace("Sample_", "")))[-1]
    num_last = int(last_sample.replace('Sample_', ''))
else:
    num_last = num_first

if num_of_predictions > 1:
    end_text = 's'
else:
    end_text = ''

print(f'\nThere were a total of {num_of_predictions} prediction{end_text}, '
      f'with {(num_last - num_first + 1) - num_of_predictions} missed')

for action, i in action_to_idx.items():
    print(f'{tracking[i]} for {action}')


if __name__ == "__main__" :
    print('\nProgramme Stopped\n')