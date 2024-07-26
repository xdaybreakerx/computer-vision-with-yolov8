# Object Detection, Tracking and Counting with YOLOv8, OpenCV, ByteTrack and Supervision

## Introduction

Ultralytics YOLOv8 is the latest version of the YOLO (You Only Look Once) object detection and image segmentation model developed by Ultralytics. 

The YOLOv8 model is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and image segmentation tasks. 

This project is designed to process video footage for detecting and counting vehicles using YOLOv8. This project can be easily modified to serve as a base for any further detection and counting you wish. 

By leveraging state-of-the-art machine learning techniques and robust video processing libraries, within this example this application provides an efficient and accurate solution for monitoring traffic flow. Whether you’re conducting traffic studies, monitoring road usage, or simply interested in vehicle detection, this tool offers a comprehensive and customizable approach. The system can detect various types of vehicles (cars, motorcycles, buses, trucks) and count them as they cross predefined lines within the video frames, making it ideal for traffic analysis and reporting.

## Accuracy and example footage
Short Example: 
[![Video Demo](./github/youtube.png)](https://www.youtube.com/watch?v=UT4w8ZaC9ZI "short video demo")

Full Length Example [here.](https://www.youtube.com/watch?v=XKMLWrsiHVg "Full length example")

As configured, with this example footage the project is highly accurate. Within the full 5 minute video, 341 objects are detected and counted. The tool's total accuracy is approximately 98.24% using YOLOv8x. 

There are the following errors:
- Left lane:
    - 3 misses: 
        - motorcycle
        - truck
        - truck 
- Middle lane:
    - 1 false positive:
        - truck 
- Right lane:
    - 2 missses: 
        - truck 

Processing the video took 2:22:27 on the Apple M2 Pro 12 core / 32 gb RAM. 

## Usage:
As configured this program is set to use the pretrained YOLOv8 model, and only detect vehicles. 

If you wish to change this you can do so by updating the class IDs within `app.py`.
```py
# Define vehicle class IDs (2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck')
selected_classes = [2, 3, 5, 7]
```

If you are unsure of the class IDs and their relevant class names for your YOLOv8 model you can use `model-names.py` to provide this to you.
```
python3 model-names.py
```

If you wish to create your own custom dataset for YOLOv8 please see [this blogpost by roboflow](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/).

## System/Hardware Requirements
- No specific hardware requirements.
- Operating System: Compatible with any OS that can run Python 3 (e.g., Windows, macOS, Linux).

## Prerequisites
- Ensure you have Python 3 installed on your system.
- Ensure ffmpeg is installed and available in your system’s PATH. You can download and install ffmpeg from [here.](https://www.ffmpeg.org/) This is only required if using the provided `youtube-dl.py` script to provide a source video to analyse, however is not needed for the core functionality of this project.


## Dependencies
The application requires the following third party Python libraries:
- `NumPy`: Used for numerical operations and handling arrays, especially in filtering detections based on class IDs. You can find more information on this library [here.](https://github.com/numpy/numpy)
- `Supervision`: Provides utilities for video processing, handling detections, object tracking, and annotating frames with bounding boxes and line zones.You can find more information on this library [here.](https://github.com/roboflow/supervision)
- `Ultralytics`: Used to load and run the YOLOv8 model for real-time object detection in video frames. You can find more information on this library [here.](https://github.com/ultralytics/ultralytics/)
- `TQDM`: Utilized to display a progress bar, providing a visual indication of the video processing progress. You can find more information on this library [here.](https://github.com/tqdm/tqdm)
- `OpenCV-python`: Used for reading and writing video files, and capturing and processing video frames. You can find more information on this library [here.](https://github.com/opencv/opencv-python)

Additionally, if used the `youtube-dl.py` requires the following:
- `yt-dlp`: Used to download videos from YouTube with specified output formats and filenames. You can find more information on this library [here.](https://github.com/yt-dlp/yt-dlp)
- `ffmpeg`: Utilized to trim the downloaded video to a specified length (30 seconds in this case). You can find more information on this library [here.](https://github.com/FFmpeg/FFmpeg)

## Installation Instructions
1. **Download the Application**: 
   Download the ZIP file of the application from the GitHub provided green ```Code``` dropdown menu, and selecting download ZIP. It is found at the upper right corner of this page.

2. **Extract the Files**: 
   Extract the contents of the ZIP file to a desired directory.

3. **Setup venv, and install requirements**:
   Open your terminal and navigate to the application's directory. Run the setup script by executing:
   ```
   bash setup.sh
   ```

   Alternatively: 
   - Create and activate virtual environment:
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    - Install requirements: 
    ```
    pip3 install -r requirements.txt 
    ```
4. **Download or specify filepath for desired video**:
    Within `app.py`, change the following lines to match the video you wish to analyse. 
    ```py    
    # Define source and target video paths
    SOURCE_VIDEO_PATH = "/path/to/file"
    TARGET_VIDEO_PATH = "/path/to/file"
    ```
    If you wish to use the video as shown in the example, run youtube-dl.py.
    ```
    python3 ./video/youtube-dl.py
    ```
5. **Change line positioning for count (optional)**:
In the provided script, lines are used to count vehicles crossing specific areas in the video. If you need to change these lines, follow the steps below:

	1.	Locate the Line Definitions:
        In the script, find the section where the line start and end points are defined. It looks like this:
    ```py 
        # Define line zones for counting
        LINE_START_ONE = sv.Point(250, 225)
        LINE_END_ONE = sv.Point(0, 175)

        LINE_START_TWO = sv.Point(375, 225)
        LINE_END_TWO = sv.Point(270, 225)

        LINE_START_THREE = sv.Point(600, 280)
        LINE_END_THREE = sv.Point(425, 230)
    ```
    2.	Modify the Line Points:
    To change the lines, you need to update the coordinates of the points. The points are defined using the sv.Point(x, y) syntax, where x and y are the coordinates on the video frame. Modify these coordinates to match the desired start and end points for your lines.
    For example, to change the first line to start at (300, 200) and end at (50, 150), update the code as follows:
    You can remove unneeded start and end points for excess lines.
    ```py
        # Define line zones for counting
        LINE_START_ONE = sv.Point(300, 200)
        LINE_END_ONE = sv.Point(50, 150)
    ```
    3.	Update the LineZone Instances:
    To modify the lines for counting, create only the desired LineZone instance(s). In the example of one line, the section should look like this after your changes:
    ```py
        # Create LineZone Instance
    line_zones = [
        sv.LineZone(start=LINE_START_ONE, end=LINE_END_ONE)
    ]
    ```
6. **Specify YOLOv8 model**: 
    Models will be downloaded on first use if not already installed. Within the script you may change the model by updating the filepath to the name of the model desired.
    
    Eg:
    ```py
    # Load YOLO Model - extra large model size
    model = YOLO("./models/yolov8x.pt")
    ```
    Can be replaced with:
    ```py
      # Load YOLO Model - nano model size
        model = YOLO("./models/yolov8n.pt")
    ```

7. **Launch the Application**:
Once the setup is complete, start the application by running:
    ```
    bash run.sh
    ```
    Alternatively: 
    ```
    python3 app.py
    ```
    If you see a progress bar appear, you are successful. 
    

## License
Distributed under the terms of the MIT License

### Thank you!
