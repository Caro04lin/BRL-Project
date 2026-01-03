# Website

The developed web site enables users to operate the differents models to predict their motions while performing painting tasks. 

<img width="1186" height="670" alt="Website" src="Web site/Prediction_interface.png" />

Clicking on the information icon displays the prequerisites. The message details the instructions about the IMU Wi-Fi configuration, the recommended use of a Wi-Fi adapter and the connexion of the IMUs and the wireless camera to the laptop.

A live feed from the wireless camera is possible, allowing the user to adjust the camera position and other users to view the current user's movements on the website.

## How to Run the Web site

1. Install **Docker Desktop** App and open it.
2. In the terminal (PowerShell for windows) after clicking on the **Web site** folder,
   - activate the virtual environment of the frontend with:
   ```bash
   docker-compose up -d
   ```
   - execute the following file to start the backend services:
   ```bash
   .\launch_backend.bat 
   ```
3. Go to Docker Desktop, then enter in website and click on the link below frontend : http://localhost:8080/

### How to use the web site for action prediction in construction painting sector

Choose one deep learning model between:
- LSTM
- GRU
- CNN-GRU
- TCN
