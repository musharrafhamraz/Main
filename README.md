
**DTreaty - Plant Disease Classification App**
**Overview**
DTreaty is a Plant Disease Classification application built using the KivyMD framework. This application allows users to capture images of plant leaves through the camera and predicts the potential disease affecting the plant. Additionally, it provides information about the detected disease and suggests treatments.

**Features**
Camera Integration: Capture images of plant leaves in real-time using the built-in camera feature.
Disease Classification: Utilizes a pre-trained deep learning model to classify plant diseases from captured images.
Alert System: Displays an alert with the detected disease and provides a button to view more details or treatment options.
Treatment Information: The application includes a Treatment screen that provides detailed information on the detected disease and possible treatments.

**Dependencies**
Python 3.x
KivyMD
TensorFlow
PIL (Pillow)
NumPy
pandas


**Installation**
Clone the repository:

bash
Copy code
**git clone https://github.com/your-username/DTreaty.git**
cd DTreaty
Install the required dependencies:

bash
Copy code
**pip install -r requirements.txt**
Run the application:

bash
Copy code
**python main.py**


**Usage**
Launch the application.
Allow camera access.
Point the camera towards a plant leaf and press the camera icon to capture an image.
The app will display an alert with the detected disease. Press "More" to view treatment information.
On the Treatment screen, detailed information about the detected disease and possible treatments will be displayed.
**Note**
Make sure to provide proper lighting conditions for accurate disease detection.
The application uses a pre-trained model (Resnet_Model.h5) for disease classification.
Contributions
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

**License**
This project is licensed under the MIT License.
