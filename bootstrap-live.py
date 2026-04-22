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

def main():
    ''' 
    Load person.png, detect pose, and save output.png
    '''
    
    # Create a pose estimation model 
    mp_pose = mp.solutions.pose
    
    # Start detecting the poses
    with mp_pose.Pose(
            static_image_mode=True,  # Set to True for static image processing
            min_detection_confidence=0.5,
            model_complexity=1) as pose:
        
        # Load test image
        image = cv2.imread("person.png")
        
        if image is None:
            print("Error: Could not load person.png. Make sure the file exists.")
            return
        
        # Make a copy for the output
        output_image = image.copy()
        
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable
        rgb_image.flags.writeable = False
        
        # Get the landmarks
        results = pose.process(rgb_image)
        
        # Convert back to writeable
        rgb_image.flags.writeable = True
        
        if results.pose_landmarks is not None:
            # Draw pose landmarks on the image
            result_image = draw_pose(output_image, results.pose_landmarks)
            
            # Save the result
            cv2.imwrite('output.png', result_image)
            print("Success! Output saved as output.png")
            print(f"Detected pose with {len(results.pose_landmarks.landmark)} landmarks")
        else:
            print('No Pose Detected in the image')
            # Save the original image anyway
            cv2.imwrite('output.png', image)
            print("Saved original image as output.png")

if __name__ == "__main__":
    main()
    print('done')
