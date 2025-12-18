from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import subprocess
import os
import sys
import re 
import signal
import cv2
import json
import threading

app = Flask(__name__)
CORS(app)

process = None #  Global process

project_dir = r"D:\Bristol Robotics Laboratory\AI Project\Multimodal-Human-Intention-Detection-for-Upper-Limb-Exoskeleton-Assistance-in-Construction-Work-main"
venv_python = os.path.join(project_dir, ".venv", "Scripts", "python.exe")

def clean_line(line):
    return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', line).strip()

@app.route("/check")
def check():
    global process
    try:
        script_path = os.path.join(project_dir, "safe_imu_data_collection.py")

        process = subprocess.Popen(
            [venv_python, "-u", script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=project_dir,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

        process.stdin.write("Y\n")
        process.stdin.flush()

        return jsonify({"output": "Start checking prerequisites "})


    except Exception as e:
            return jsonify({"output": f"Error: {str(e)}"})


@app.route("/start_recording")
def start_recording():
    global process

    if process is None:
        return jsonify({"output": "Error: Process not running."})

    try:
        process.stdin.write("\n")
        process.stdin.flush()

        return jsonify({"output": "Recording command sent."})

    except Exception as e:
        return jsonify({"output": f"Error: {str(e)}"})


@app.route("/pause", methods=["POST"])
def pause():
    global process
    if process is None:
        return jsonify({"output": "Error: no process."})

    try:
        process.stdin.write(" ")
        process.stdin.flush()
        return jsonify({"output": "Paused/Resumed"})

    except Exception as e:
        return jsonify({"output": f"Error: {str(e)}"})

@app.route("/stream")
def stream_output():
    global process

    if process is None:
        return Response("data: ERROR: process not running\n\n",
                        mimetype="text/event-stream")

    def generate():
        while process and process.poll() is None: # process still alive?
            line = process.stdout.readline()
            if not line:
                break  # Process ended

            clean = clean_line(line)
            if clean:
                yield f"data: {clean}\n\n"
         # When process dies → tell frontend to close stream
        yield "data: STREAM_CLOSED\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.route("/stop", methods=["POST"])
def stop():
    global process

    if process is None:
        return jsonify({"output": "Error: no process running."})

    try:
        # Send CTRL+C event
        process.send_signal(signal.CTRL_C_EVENT)

        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill() 

        process = None  

        return jsonify({"output": "Process stopped"})

    except Exception as e:
        return jsonify({"output": f"Error: {str(e)}"})
    

@app.route("/camera_feed")
def camera_feed():
    camera_url = "rtsp://192.168.1.1:554/live.sdp"
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        return Response("Camera not accessible", status=500)

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            # Stream frame by frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

model_scripts = {
    "LSTM": "safe_Local_Prediction.py",
    "TCN": "TCN_Local_Prediction_Final.py",
    "CNN-GRU": "CNN_GRU_Local_Prediction_2.py",
    "GRU": "GRU_Local_Prediction.py"
}

# Store processes for each model
model_processes = {}

@app.post("/run_model_stream")
def run_model_stream():
    data = request.get_json()
    models = data.get("models", [])
    started = []

    for model in models:
        if model not in model_scripts:
            continue

        script_path = os.path.join(project_dir, model_scripts[model])

        process = subprocess.Popen(
            [venv_python, "-u", script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=project_dir,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

        model_processes[model] = process
        started.append(model)

        try:
            process.stdin.write("N\n")
            process.stdin.flush()
        except:
            pass

    return jsonify({"started": started})


@app.get("/model_stream/<model_name>")
def model_stream(model_name):
    if model_name not in model_processes:
        return Response("data: ERROR: Model not running\n\n",
                        mimetype="text/event-stream")

    process = model_processes[model_name]

    def generate():
        for line in iter(process.stdout.readline, ""):
            clean = line.strip()
            if not clean:
                continue

            # Send raw line to terminal
            yield f"data: {clean}\n\n"

            # Try to parse as JSON for results table
            try:
                json_output = json.loads(clean)
                yield f"event: data_json\ndata: {json.dumps(json_output)}\n\n"
            except:
                pass

        # Process ended
        yield "data: STREAM_CLOSED\n\n"

    return Response(generate(), mimetype="text/event-stream")

@app.post("/stop_model/<model_name>")
def stop_model(model_name):
    if model_name in model_processes:
        try:
            process = model_processes[model_name]
            process.send_signal(signal.CTRL_C_EVENT) # CTRL+C
        except:
            pass

        del model_processes[model_name]

    return {"stopped": model_name}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)





