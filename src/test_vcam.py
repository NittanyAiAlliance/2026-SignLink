# test_vcam.py - Simple test to verify virtual camera works

import cv2
import pyvirtualcam
import numpy as np

print("Testing virtual camera...")
print("=" * 60)

# Open your webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Cannot open webcam")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

print(f"✅ Webcam opened: {width}x{height}")
print(f"Starting virtual camera...")

try:
    with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        print(f"✅ Virtual camera started: {cam.device}")
        print(f"   Resolution: {width}x{height} @ {fps}fps")
        print("\nNow open WhatsApp/Zoom and select the virtual camera!")
        print("You should see yourself with 'TEST' written on screen")
        print("Press 'q' to quit\n")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add "TEST" text overlay
            cv2.putText(frame, "TEST - Virtual Camera Working!", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Send to virtual camera
            cam.send(frame)
            
            # Show preview
            cv2.imshow("Preview (what others see)", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            cam.sleep_until_next_frame()
            
except KeyboardInterrupt:
    print("\n\n⚠️ Interrupted by user")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test complete")