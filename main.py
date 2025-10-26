"""
Simple Face Anonymization Application
Works with webcam or video files
Press 'q' to quit
"""

import cv2
import sys

# Load face detection classifier from local directory
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def blur_faces(frame):
    """Detect and blur faces in a frame"""
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    # Blur each detected face
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_face
    
    # Display face count
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def main():
    """Main function"""
    # Check if video file is provided as argument
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        print(f"Using video file: {video_source}")
    else:
        video_source = 0  # Use webcam
        print("Using webcam (to use video file: python main.py video.mp4)")
    
    # Open video source
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        sys.exit(1)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or error reading frame")
            break
        
        # Process frame
        processed_frame = blur_faces(frame)
        
        # Show result
        cv2.imshow('Face Anonymization - Press q to quit', processed_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
