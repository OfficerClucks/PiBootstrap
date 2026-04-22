# Imports
import mediapipe as mp
import time
import cv2

# Initialize the USB camera (0 is usually the first USB camera)
video_capture = cv2.VideoCapture(0)

# Check if camera opened successfully
if not video_capture.isOpened():
    print("Error: Could not open USB camera")
    exit(1)

# Set lower resolution for better performance on Pi 3B
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Give the camera a moment to warm up
time.sleep(1)

def draw_pose(image, landmarks):
    ''' 
    TODO Task 1
    
    Code to this fucntion to draw circles on the landmarks and lines
    connecting the landmarks then return the image.
    
    Use the cv2.line and cv2.circle functions.

    landmarks is a collection of 33 dictionaries with the following keys
        x: float values in the interval of [0.0,1.0]
        y: float values in the interval of [0.0,1.0]
        z: float values in the interval of [0.0,1.0]
        visibility: float values in the interval of [0.0,1.0]
        
    References:
    https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
    https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    '''

    # copy the image
    landmark_image = image.copy()
    
    # get the dimensions of the image
    height, width, _ = image.shape
    
    # Define connections between landmarks (POSE_CONNECTIONS)
    # These are the standard MediaPipe pose connections
    connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        # Arms
        (9, 10), (11, 12), (11, 13), (13, 15), 
        (12, 14), (14, 16), (15, 17), (15, 19), 
        (15, 21), (17, 19), (16, 18), (16, 20), 
        (16, 22), (18, 20), (11, 23), (12, 24),
        # Legs
        (23, 24), (23, 25), (24, 26), (25, 27), 
        (26, 28), (27, 29), (28, 30), (27, 31), 
        (28, 32), (29, 31), (30, 32)
    ]
    
    # Draw connections (lines)
    for connection in connections:
        start_idx, end_idx = connection
                
        # Get landmark coordinates
        start_landmark = landmarks.landmark[start_idx]
        end_landmark = landmarks.landmark[end_idx]
        
        # Convert normalized coordinates to pixel coordinates
        start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
        end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
        
        # Draw line if visibility is high enough
        if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
            cv2.line(landmark_image, start_point, end_point, (0, 255, 0), 2)
    
    # Draw circles on landmarks
    for idx, landmark in enumerate(landmarks.landmark):
        # Only draw if visibility is high enough
        if landmark.visibility > 0.5:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(landmark_image, (x, y), 4, (0, 0, 255), -1)
    
    return landmark_image

def main():
    ''' 
    TODO Task 2 & 3: Modified to use USB camera with video feed
    '''

    # Create a pose estimation model 
    mp_pose = mp.solutions.pose
    
    # For performance metrics
    prev_time = 0
    
    # start detecting the poses
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1) as pose:

        print("Starting pose detection. Press 'q' to quit, 's' to save snapshot")
        
        while True:
            # Read a frame from the USB camera
            ret, image = video_capture.read()
            
            if not ret:
                print("Failed to capture image from USB camera")
                break
            
            # Flip the image horizontally for mirror view (optional)
            image = cv2.flip(image, 1)
            
            # Make a copy for display
            display_image = image.copy()
            
            # Convert BGR to RGB (MediaPipe uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # To improve performance, mark the image as not writeable
            rgb_image.flags.writeable = False
            
            # Get the landmarks
            results = pose.process(rgb_image)
            
            # Convert back to BGR for OpenCV
            rgb_image.flags.writeable = True
            
            if results.pose_landmarks != None:
                # Draw pose landmarks on the image
                result_image = draw_pose(display_image, results.pose_landmarks)
                
                # Calculate and display FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
                prev_time = current_time
                
                # Display FPS on image
                cv2.putText(result_image, f"FPS: {int(fps)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)                
                # Show instructions
                cv2.putText(result_image, "Press 'q' to quit, 's' for snapshot", 
                           (10, result_image.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the result
                cv2.imshow('Pose Detection - USB Camera', result_image)
            else:
                # Show message when no pose detected
                cv2.putText(display_image, "No Pose Detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 0, 255), 2)
                cv2.imshow('Pose Detection - USB Camera', display_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save snapshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"pose_snapshot_{timestamp}.png"
                if results.pose_landmarks != None:
                    cv2.imwrite(filename, result_image)
                else:
                    cv2.imwrite(filename, display_image)
                print(f"Snapshot saved as {filename}")

    # Release the camera and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    print('Done!')

if __name__ == "__main__":
    main()
