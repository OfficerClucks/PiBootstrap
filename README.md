# Imports
import mediapipe as mp
import cv2
import time

def draw_pose(image, landmarks):
    ''' 
    Draw circles on the landmarks and lines connecting the landmarks
    
    Uses cv2.line and cv2.circle functions.
    
    landmarks is a MediaPipe PoseLandmarker output with 33 landmarks
    '''
    
    # Copy the image
    landmark_image = image.copy()
    
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Define connections between landmarks (standard MediaPipe pose connections)
    connections = [
        # Face connections
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        # Arms
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (11, 23), (12, 24),
        # Body
        (23, 24),
        # Legs
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
    ]
    
    # Draw connections (lines)
    for connection in connections:
        start_idx, end_idx = connection
        
        # Get landmark coordinates
        start_landmark = landmarks.landmark[start_idx]
        end_landmark = landmarks.landmark[end_idx]
        
        # Check visibility (only draw if both points are visible enough)
        if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
            # Convert normalized coordinates to pixel coordinates
            start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
            end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
            
            # Draw line (green, thickness 2)
            cv2.line(landmark_image, start_point, end_point, (0, 255, 0), 2)
    
    # Draw circles on landmarks
    for idx, landmark in enumerate(landmarks.landmark):
        # Only draw if visibility is high enough
        if landmark.visibility > 0.5:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            # Draw circle (red, radius 4, filled)
            cv2.circle(landmark_image, (x, y), 4, (0, 0, 255), -1)
    
    return landmark_image

def capture_photo():
    ''' 
    Capture a single photo using the USB camera
    Returns the captured image or None if failed
    '''
    
    # Initialize the USB camera
    video_capture = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open USB camera")
        return None
    
    # Set resolution for better performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Give the camera a moment to warm up
    time.sleep(1)
    
    print("Preparing to take photo...")
    
    # Show preview for 2 seconds to let person get ready
    print("Getting ready...")
    for i in range(3, 0, -1):
        ret, frame = video_capture.read()
        if ret:
            # Display countdown on frame
            cv2.putText(frame, f"Get ready... {i}", 
                       (frame.shape[1]//2 - 100, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Countdown', frame)
            cv2.waitKey(1000)
    
    # Take the photo
    print("Taking photo...")
    ret, image = video_capture.read()
    
    if not ret:
        print("Error: Failed to capture image from USB camera")
        video_capture.release()
        cv2.destroyAllWindows()
        return None
    
    # Optional: Flip the image horizontally for mirror view
    image = cv2.flip(image, 1)
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    
    print("Photo captured successfully!")
    return image

def main():
    ''' 
    Capture a photo from USB camera, detect pose, and save output.png
    '''
    
    # Capture photo from camera
    image = capture_photo()
    
    if image is None:
        print("Failed to capture photo. Exiting.")
        return
    
    # Create a pose estimation model 
    mp_pose = mp.solutions.pose
    
    # Start detecting the poses
    with mp_pose.Pose(
            static_image_mode=True,  # Set to True for static image processing
            min_detection_confidence=0.5,
            model_complexity=1) as pose:
        
        # Make a copy for the output
        output_image = image.copy()
        
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, mark the image as not writeable
        rgb_image.flags.writeable = False
        
        # Get the landmarks
        results = pose.process(rgb_image)
        
        # Convert back to writeable
        rgb_image.flags.writeable = True
        
        if results.pose_landmarks is not None:
            # Draw pose landmarks on the image
            result_image = draw_pose(output_image, results.pose_landmarks)
            
            # Add text indicating successful detection
            cv2.putText(result_image, "Pose Detected!", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            # Save the result
            cv2.imwrite('output.png', result_image)
            print("Success! Output saved as output.png")
            print(f"Detected pose with {len(results.pose_landmarks.landmark)} landmarks")
            
            # Also display the image briefly
            cv2.imshow('Pose Detection Result', result_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('No Pose Detected in the captured photo')
            # Add text indicating no pose detected
            cv2.putText(output_image, "No Pose Detected - Please try again", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            # Save the original image anyway
            cv2.imwrite('output.png', output_image)
            print("Saved original image as output.png")
            cv2.imshow('No Pose Detected', output_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    print('done')
