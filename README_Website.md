# Website

The developed web site enables users to operate the differents models to predict their motions while performing painting tasks. 

<img width="1186" height="670" alt="Website" src="Web site/Prediction_interface.png" />

Clicking on the information icon displays the prequerisites. The message details the instructions about the IMU Wi-Fi configuration, the recommended use of a Wi-Fi adapter and the connexion of the IMUs and the wireless camera to the laptop. When all prequesites are completed, the battery levels of the five IMUs and their estimated autonomy are displayed, preventing unexpected disconnections during motion detection.

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
3. Go to Docker Desktop, then enter in website and click on the link below frontend : `http://localhost:8080/`

### How to use the web site for action prediction 

1. Choose one deep learning model between:
- **LSTM**
- **GRU**
- **CNN-GRU**
- **TCN**

2. Ensure all prerequisites are completed where instructions are provided in the **information icon**.
3. Click on the button **Check Prerequisites**, which verify that all conditions were filled.
   Option: Click on the button **Display the camera's screen** to see the screen of the wireless camera.
4. Click on the button **Start the Collecting Data** to start motion prediction. Prediction can be paused or stopped using the **Pause** and **Stop** buttons. If multiple models are runned at the same time, the predictions can be stopped with the **Stop all models** button. 


