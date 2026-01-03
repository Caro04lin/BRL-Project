# Upper Limb Exoskeleton Real-Time Detection

This project provide four **real-time human intention recognition system** for an upper limb exoskeleton. 
Each system include a pre-trained model **YOLO** model for detecting painting tools and hands. MoViNet was selected to recognize user actions based on video frames. Multiple deep learning architectures, including the Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), Convolutional Neural Network–Gated Recurrent Unit (CNN-GRU) and Temporal Convolutional Network (TCN) models, were developed, trained, and compared to get the most accurate model for real-time detection of user intent during painting tasks, including the tool being used and the current task phase. 

<img width="1186" height="670" alt="Fusion" src="Multimodal-Human-Intention-Detection-for-Upper-Limb-Exoskeleton-Assistance-in-Construction-Work-main/Multimodal_fusion_architecture.png" />

Deep learning models are trained on a custom Database (link:).
The system outputs:

Action label: one of 10 classes (e.g., Bimanual_Up, Bimanual_Right, Unimanual_Down).
Detected tool: Brush, Short roller, or Long roller.


