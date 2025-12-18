if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

import csv                      # For csv writing
import os                       # To manage folders and paths
import sys                      # For quitting program early
import math                     # To calculate elbow angle(in Print function)
import threading
import keyboard
import pandas as pd
import numpy as np
from time import sleep, time

# ----   # Modifiable variables   ----
Script_dir = os.path.dirname(os.path.abspath(__file__))  # folder "Multimodal-Human-Intention-Detection-for-Upper-Limb-Exoskeleton-Assistance-in-Construction-Work-main" 
root_directory = os.path.join(Script_dir, "Temporary_Data")    # Directory where temporary folders are stored
Ask_cam_num: bool =     False               # Set to True to ask the user to put the cam number themselves, if False, default is set below
cam_num: int =          0                   # Set to 0 to activate the camera, but 1 if yoy have a builtin camera
NEW_CAM : bool =        True               # Set to True if you are using the new camera
fps: int =              30                  # Number of save per seconds
buffer: int =           1500                # Number of folders saved
CleanFolder: bool =     False               # If True, delete all temporary folders at the end
wifi_to_connect_1: str =  'Upper_Limb_Exo'    # The Wi-Fi where the raspberry pi and IMUs are connected
wifi_to_connect_2: str = 'EP6_PLUS_401229'  # The wifi of the wireless camera : Ordro EP6 Plus
window_size: int =      200                  # How many lines of IMU data will be displayed at the same time
PRINT_IMU =             True                # If true print the imu data in the terminal
IMU_IDS = ['68592362', '68591F90', '685928A2', '68592647', '68B03E54'] 
# IMU:  LEH, LES, LB, REH, RES
# IMU X-DIR: L H<-E->S, B H->F, R H<-E->S
# ------------------------------------


try :
    import cv2      # For the camera
    import ximu3    # For the IMU
    from pupil_labs.realtime_api.simple import discover_one_device
    import pandas as pd
except ModuleNotFoundError as Err :
    missing_module = str(Err).replace('No module named ','')
    missing_module = missing_module.replace("'",'')
    if missing_module == 'cv2' :
        sys.exit(f'No module named {missing_module} try : pip install opencv-python')
    elif missing_module == "pupil_labs" :
        sys.exit(f'No module named {missing_module} try : pip install pupil-labs-realtime-api')
    else :
        print(f'No module named {missing_module} try : pip install {missing_module}')

try :
    from Imports.Safe_Functions import format_time, connected_wifi, ask_yn, calculate_angle_between_vectors, extract_arm_direction_angles, process_imu_to_new
except ModuleNotFoundError :
    sys.exit('Missing Import folder, make sure you are in the right directory')

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


try :
    if Ask_cam_num :
        NEW_CAM = ask_yn('\033cAre you using the new camera ?(Y/N) ')
        if  not NEW_CAM:
            cam_num = int(input("Cam Number : "))
        if cam_num < 0: 
            raise ValueError
except (ValueError, TypeError) :
    sys.exit("Invalid Cam Number") 
except KeyboardInterrupt :
    sys.exit("\n\nProgramme Stopped\n")

# We check if the root directory exist
if not os.path.exists(root_directory) :
    os.makedirs(root_directory)
elif os.listdir(root_directory):  # If there are files in the directory : True
    if ask_yn(f'\033c{root_directory} not empty do you want to clear it ? (Y/N)') :
        print('Clearing ...')
        for folders_to_del in os.listdir(root_directory):
            for files_to_del in os.listdir(f"{root_directory}/{folders_to_del}"):
                os.remove(os.path.join(f'{root_directory}/{folders_to_del}', files_to_del))
            os.rmdir(f"{root_directory}/{folders_to_del}")
    elif ask_yn('Do you want to save it ? (Y/N)') :
        Folder_Name = str(input("Folder Name : "))
        if root_directory != Folder_Name and Folder_Name != '' :
            os.rename(root_directory, Folder_Name)
        else : sys.exit("Incorrect Folder Name")
    else : sys.exit('Cannot access non-empty folder, Programme Stopped\n')

print("\033cStarting ...\n") # Clear Terminal
print("Checking Wifi ...")
print(LINE_UP, end=LINE_CLEAR)

success, connected_list = connected_wifi()

if success : 
    print(f'Detected active Wifi connections: {connected_list}')
    if (wifi_to_connect_1 in connected_list or wifi_to_connect_1 + "_5G" in connected_list) and \
        (wifi_to_connect_2 in connected_list):
        print("Both required Wifi networks are connected.")
    else :
        print("Missing one or both required Wi-Fi networks.")
        print(f"Expected: [{wifi_to_connect_1},{wifi_to_connect_1 + '_5G'}, {wifi_to_connect_2}]")
        sys.exit()
else:
    print("Could not check Wi-Fi connections.")

# Initialize sensor values to 0
imu_data = {
    imu_id: {
        "gyr": [0, 0, 0],
        "acc": [0, 0, 0],
        "rotm": [[0.0] * 3 for _ in range(3)],
        "battery": None
    } for imu_id in IMU_IDS
}

is_paused = False  # press "space" button then pause  the datacollection

class Connection:
    def __init__(self, connection_info):
        self.__connection = ximu3.Connection(connection_info)
        if self.__connection.open() != ximu3.RESULT_OK:
            sys.exit("Unable to open connection " + connection_info.to_string())
        ping_response = self.__connection.ping()
        self.__prefix = ping_response.serial_number
        if ping_response.result != ximu3.RESULT_OK:
            print("Ping failed for " + connection_info.to_string())
            raise AssertionError
        self.__connection.add_inertial_callback(self.__inertial_callback)
        self.__connection.add_rotation_matrix_callback(self.__rotation_matrix_callback)
        self.__connection.add_battery_callback(self.__battery_callback)


    def __battery_callback(self, msg):
        """Callback when IMU sends a battery update"""
        if self.__prefix in imu_data:
            imu_data[self.__prefix]["battery"] = msg.percentage

    def close(self):
        self.__connection.close()

    def send_command(self, key, value=None):
        if value is None:
            value = "null"
        elif type(value) is bool:
            value = str(value).lower()
        elif type(value) is str:
            value = "\"" + value + "\""
        else:
            value = str(value)

        command = "{\"" + key + "\":" + value + "}"

        responses = self.__connection.send_commands([command], 2, 500)

        if not responses:
            sys.exit("Unable to confirm command " + command + " for " + self.__connection.get_info().to_string())
        else:
            print(self.__prefix + " " + responses[0])

    def __inertial_callback(self, msg):
        if self.__prefix in imu_data:
            imu_data[self.__prefix]["gyr"] = [msg.gyroscope_x, msg.gyroscope_y, msg.gyroscope_z]
            imu_data[self.__prefix]["acc"] = [msg.accelerometer_x, msg.accelerometer_y, msg.accelerometer_z]

    def __rotation_matrix_callback(self, msg):
        if self.__prefix in imu_data:
            imu_data[self.__prefix]["rotm"] = [
                [msg.xx, msg.xy, msg.xz],
                [msg.yx, msg.yy, msg.yz],
                [msg.zx, msg.zy, msg.zz]
            ]

def toggle_pause():
    """Toggle the pause state and print the status immediately."""
    global is_paused
    is_paused = not is_paused
    print("\n[PAUSED]" if is_paused else "\n[RESUMED]", flush=True)

def key_listener_keyboard():
    """Detect real keyboard SPACE key."""
    global is_paused
    while True:
        keyboard.wait('space')
        toggle_pause()

def key_listener_stdin():
    """Detect space or enter sent by backend through stdin."""
    global is_paused
    while True:
        ch = sys.stdin.read(1)  # blocking read
        if not ch:
            break  # stdin closed by parent

        if ch == ' ':
            toggle_pause()

# Establish connections
print("Checking IMU connections...")
while True:
    try:
        connections = [Connection(m.to_udp_connection_info()) for m in ximu3.NetworkAnnouncement().get_messages_after_short_delay()]
        break
    except AssertionError:
        pass
if not connections:
    print(LINE_UP, end=LINE_CLEAR)
    sys.exit("No UDP connections to IMUs")
print(LINE_UP, end=LINE_CLEAR)
print('Connected to IMUs')

# Video capture setup
print("Checking camera ...")

if NEW_CAM:
    # (Last version for the previous camera) Look for devices. Returns as soon as it has found the first device.
    # device = discover_one_device(max_search_duration_seconds=10)
    # if device is None:
    #     print(LINE_UP, end=LINE_CLEAR)
    #     sys.exit("No device found.")

    # print(LINE_UP, end=LINE_CLEAR)
    # print(f"Connected to {device}")

    cam_message = 'Using New Camera \n' 
    stream_url = f"rtsp://192.168.1.1:554/live.sdp"  # Ordro EP6 Plus default RTSP URL
else :
    # Previous camera (USB)
    cap = cv2.VideoCapture(cam_num)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ret, frame = cap.read()
    if not ret: # If camera is unavailable :
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        for connection in connections:
            connection.close()
        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)
        sys.exit('Camera disconnected.')
    cam_message = f'Camera Number : {cam_num} \n'

print(LINE_UP, end=LINE_CLEAR)
print('Connected to Camera\n')

print("Waiting for battery updates...")
sleep(5)  # give IMUs time to send telemetry packets

def estimate_autonomy(battery_percent, full_hours):
    total_minutes = battery_percent / 100 * full_hours * 60
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return f"{hours}h {minutes:02d}min"

if wifi_to_connect_1 + "_5G" in connected_list:
    wifi_band_hours = 6
    wifi_band_label = "5 GHz"
else:
    wifi_band_hours = 9
    wifi_band_label = "2.4 GHz"

print(f"\033[1A\033[KIMUs Batteries and Autonomy ({wifi_band_label} mode):\n") 

for imu_id in IMU_IDS:
    battery = imu_data[imu_id]["battery"]
    if battery is not None:
        battery_value = round(battery)
        autonomy = estimate_autonomy(battery_value, wifi_band_hours)
        if battery_value <= 10:
            alert = "⚠️  Critical battery"
        elif battery_value <= 20:
            alert = "⚠️  Low battery"
        else:
            alert = ""
        if alert:
            print(f"IMU {imu_id}: {battery:.0f}% ({autonomy}) {alert}")
        else:
            print(f"IMU {imu_id}: {battery:.0f}% ({autonomy})" )
    else:
        print(f"IMU {imu_id}: not received yet")


try :
    input('\nProgramme Ready, Press Enter to Start')
    for i in range(2) :
        print(f'Starting in {2-i}s')
        sleep(1)
        print(LINE_UP, end=LINE_CLEAR)
except KeyboardInterrupt :
    sys.exit('\nProgramme Stopped\n')

input("\nReady. Press Enter to start recording...\n")

# Open camera only now
if NEW_CAM:
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        sys.exit("Error: Unable to access the camera at recording start.")
    print("Camera stream started.")

sequence_length = 10    # Size of samples default 10
sample_counter = 0
frames_counter = 0
Start_Time = time()
print("Recording started!\n")

recording_started_time = time()
first_battery_display_done = False

def monitor_imu_battery(imu_data, IMU_IDS):
    """
    Monitors IMU battery in real-time.
    Adjusts check interval based on battery level.
    Displays alerts for low and critical batteries.
    """
    global recording_started_time, first_battery_display_done
    last_check = {imu_id: 0 for imu_id in IMU_IDS}

    while True:
        current_time = time()
        
        # --- FIRST BATTERY DISPLAY 5 MIN AFTER RECORDING STARTED ---
        if not first_battery_display_done:
            if current_time - recording_started_time >= 300:  # 5 min
                print("\n--- IMU Battery Status ---\n")
                for imu_id in IMU_IDS:
                    battery = imu_data[imu_id]["battery"]
                    if battery is None:
                        continue

                battery_value = round(battery)

                # Decide interval based on battery level
                if battery_value >= 60:
                    interval = 20 * 60  # 20 minutes
                elif battery_value >= 35:
                    interval = 10 * 60  # 10 minutes
                else:
                    interval = 5 * 60   # 5 minutes

                # Only check if interval has passed
                if current_time - last_check[imu_id] >= interval:
                    last_check[imu_id] = current_time

                # Alert logic
                alert_msg = ""
                if battery_value <= 10:
                    alert_msg = "⚠️  Critical battery"
                elif battery_value <= 20:
                    alert_msg = "⚠️  Low battery"

                # Print battery status
                autonomy = estimate_autonomy(battery_value, wifi_band_hours)
                
                if alert_msg:
                    print(f"IMU {imu_id}: {battery_value:.0f}% ({autonomy}) {alert_msg}")
                    print('\a', end='')  # optional beep
                else:
                    print(f"IMU {imu_id}: {battery_value:.0f}% ({autonomy})")
                
        
            sleep(1)  # small sleep to prevent busy loop
            continue

        for imu_id in IMU_IDS:
                battery = imu_data[imu_id]["battery"]
                if battery is None:
                    continue

                battery_value = round(battery)

                # Decide interval based on battery level
                if battery_value >= 60:
                    interval = 20 * 60  # 20 minutes
                elif battery_value >= 35:
                    interval = 10 * 60  # 10 minutes
                else:
                    interval = 5 * 60   # 5 minutes

                # Only check if interval has passed
                if current_time - last_check[imu_id] >= interval:
                    last_check[imu_id] = current_time

                # Alert logic
                alert_msg = ""
                if battery_value <= 10:
                    alert_msg = "⚠️  Critical battery"
                elif battery_value <= 20:
                    alert_msg = "⚠️  Low battery"

                # Print battery status
                autonomy = estimate_autonomy(battery_value, wifi_band_hours)
                
                if alert_msg:
                    print(f"IMU {imu_id}: {battery_value:.0f}% ({autonomy}) {alert_msg}")
                    print('\a', end='')  # optional beep
                else:
                    print(f"IMU {imu_id}: {battery_value:.0f}% ({autonomy})")    

        sleep(1)  # small sleep to prevent busy loop


# start read from keyboard input
threading.Thread(target=monitor_imu_battery, args=(imu_data, IMU_IDS), daemon=True).start()
threading.Thread(target=key_listener_keyboard, daemon=True).start()
threading.Thread(target=key_listener_stdin, daemon=True).start() # Listen to stdin for pause commands from backend
print("Press SPACE to pause/resume.")

try : # try except is to ignore the keyboard interrupt error
    message = f'Programme running   ctrl + C to stop\n\nClean Folder : {CleanFolder} \n' + cam_message
    print('\033c'+message)
    while True:
        if is_paused:
            sleep(0.1)
            continue

        sample_counter += 1
        # We create a folder with a csv file in it( csv with rotation matrix)
        folder = f"{root_directory}/Sample_{sample_counter}"
        os.makedirs(folder)
        #csv_file = open(f"{folder}/imu.csv", 'w', newline='')
        with open(f"{folder}/imu.csv", 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            #based on that calculate new csv with joint angle
            #csv_file.close()

            for i in range(sequence_length):
                frames_counter += 1
                while time() - Start_Time < frames_counter / fps:
                    sleep(0.001)

                row = []
                for imu_id in IMU_IDS:
                    row += imu_data[imu_id]["gyr"] + imu_data[imu_id]["acc"] + sum(imu_data[imu_id]["rotm"], [])
                writer.writerow(row)

                if NEW_CAM :
                    # Read frame from the new camera (Ordro EP6 Plus)
                    ret, frame = cap.read() 
   #Previous cam    bgr_pixels, frame_datetime = device.receive_scene_video_frame()
                    # ret = 
   #Previous cam    frame = bgr_pixels # TODO Possible source of error, check conversion
                    if not ret: # If camera is unavailable :
                        # Release resources
                        cap.release()
                        cv2.destroyAllWindows()
                        csv_file.close()        
                        for connection in connections:
                            connection.close()
                        print('\nCamera disconnected')
                        raise KeyboardInterrupt
                else :
                    ret, frame = cap.read()
                    if not ret: # If camera is unavailable :
                        # Release resources
                        cap.release()
                        cv2.destroyAllWindows()
                        csv_file.close()
                        for connection in connections:
                            connection.close()
                        print('\nCamera disconnected')
                        raise KeyboardInterrupt

                if PRINT_IMU:
                    #print(f"Frame {frames_counter}, Sample {sample_counter}")
                    if frames_counter%window_size == 0 :
                        print('\033c'+message)

                # Add image
                
                cv2.imwrite(f"{folder}/frame_{frames_counter}.jpg", frame)

        #csv_file.close()
        process_imu_to_new(folder)

        # We delete the folders as we go so that we don't saturate
        if sample_counter > buffer:
            del_folder = f"{root_directory}/Sample_{sample_counter - buffer}"
            for file in os.listdir(del_folder):
                os.remove(os.path.join(del_folder, file))
            os.rmdir(del_folder)

except KeyboardInterrupt:
    t = round(time() - Start_Time, 4)
    print(f"\nStopped after {frames_counter} frames in {format_time(t)} — FPS: {frames_counter / t:.2f}")
    try:
        csv_file.close()
    except:
        pass
    if CleanFolder:
        for folders_left in os.listdir(root_directory) :
            for files_left in os.listdir(f"{root_directory}/{folders_left}"):
                os.remove(os.path.join(f'{root_directory}/{folders_left}', files_left))
            os.rmdir(f"{root_directory}/{folders_left}")
        os.rmdir(root_directory)
        if not NEW_CAM:
            cap.release()
        else:
            device.close()
        for c in connections:
            c.close()
        cv2.destroyAllWindows()


if __name__ == "__main__" :
    print('\nProgramme Stopped\n')