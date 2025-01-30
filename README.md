# Yoga Pose Correction

This project utilizes computer vision and machine learning techniques to assist yoga practitioners in achieving correct postures. By leveraging OpenCV and MediaPipe, the system analyzes body movements and provides real-time feedback to improve yoga poses.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Yoga Pose Correction project captures video input, detects body landmarks, analyzes poses, and offers corrective feedback to users. This approach aids practitioners in refining their yoga postures through real-time analysis.

## Technologies Used

- **OpenCV**: An open-source computer vision library used for image and video processing.
- **MediaPipe**: A framework for building multimodal applied machine learning pipelines, providing pre-trained models for pose estimation.

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/pranav-582/yoga_pose_correction.git
   cd yoga_pose_correction
   ```

2. **Install Dependencies:**

   Ensure you have Python installed. Then, install the required libraries:

   ```bash
   pip install opencv-python mediapipe
   ```

## Usage

1. **Run the Application:**

   Execute the main script to start the pose correction application:

   ```bash
   python yoga_pose_correction.py
   ```

2. **Provide Video Input:**

   The system captures video from your webcam or a specified video file. Ensure your webcam is functional or provide a valid video file path.

3. **Receive Feedback:**

   As you perform yoga poses, the system analyzes your posture and provides real-time feedback to help you adjust and improve your form.

## Workflow

1. **Video Capturing:**

   The system starts by capturing video from a webcam or a video file using OpenCV.

2. **Pose Detection with MediaPipe:**

   MediaPipe's pose detection solution identifies and tracks body landmarks (e.g., shoulders, elbows, hips, knees) in real-time.

3. **Processing Frames:**

   Each frame from the video is processed to detect poses and analyze the correctness of the posture.

4. **Providing Feedback:**

   Based on the analysis, the system offers corrective suggestions to help users achieve proper alignment in their yoga poses.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

This README provides an overview of the Yoga Pose Correction project, including setup instructions and usage guidelines. For detailed code implementation and further explanations, please refer to the project's source code and accompanying documentation. 
