# Computer-Vision-Final-Class-Project

This project aimed to develop a robust and accurate traffic detection and forecasting system using computer vision techniques. However, due to time constraints, a fully robust system was not able to be built. The Haar Cascade Classifier and YOLOv4 model were implemented to detect cars and trucks in video feeds from GDOT traffic cameras. While the YOLOv4 model demonstrated superior performance in detecting both cars and trucks, there is still room for improvement by training the model on the specific GA511 video data. The Haar Cascade Classifier, although computationally more efficient, also requires further training on the specific data to be viable for real-time traffic analysis. However, based on the preliminary results gained from this project, the YOLO model is recommended for traffic analysis applications requiring higher accuracy and the ability to detect multiple object classes. The Haar Cascade Classifier can be considered for simpler use cases or when computational resources are limited. But again, both models would require extensive training for specific traffic data from the state of Georgia, to achieve more satisfactory detection results.

The results of this project indicate the potential of computer vision techniques in traffic detection and forecasting. In future work, further training, and optimization of the models on the specific GA511 video data could lead to significant improvements in detection performance. Improvements could also be made in the preprocessing of the data, such as background subtraction to further isolate only the vehicles, which could lead to better detection accuracy. Additionally, the system could be expanded to cover more camera sites throughout the state and integrated with other GDOT systems to improve accident response time, roadwork and infrastructure planning, and traffic monitoring for better deployment of resources. Ultimately, the application of these computer vision techniques has the potential to significantly impact traffic management and enhance overall traffic safety and efficiency.
