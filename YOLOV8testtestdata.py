from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
model=YOLO('yolov8m-pose.pt')
import tensorflow
tf_model = tensorflow.keras.models.load_model('mmodelfinal1.h5')
actions = np.array(['mixue','em']) #all the actions classes in sequence
list=[]
n=0
poselist=[]

cap = cv2.VideoCapture("WIN_20240809_17_35_03_Pro.mp4")
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height) 
pose_name="chuabiet"
writer=cv2.VideoWriter('testmixue.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        for r in results:
        # Visualize the results on the frame
         keypoints=r.keypoints 
         B=np.array(keypoints.data)
         A=B[0].reshape(51)
         list.append(   np.array(A))
         print(len(list))
         font = cv2.FONT_HERSHEY_SIMPLEX 
         cv2.putText(frame,  
                pose_name,  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
         if (len(list)==40):
           result = tf_model.predict(np.expand_dims(list, axis=0))
           pose_name = actions[np.argmax(result)]
           poselist.append(pose_name)
           print(pose_name)
           font = cv2.FONT_HERSHEY_SIMPLEX 
           if(pose_name=="mixue"):
            cv2.putText(frame,  
                pose_name,  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
           if(pose_name=="em"):
            cv2.putText(frame,  
                pose_name,  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
           else:
               cv2.putText(frame,  
                'khongbiet',  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
            
           list=[]
           print(len(list))
         annotated_frame = results[0].plot()
         writer.write(annotated_frame)
        # Display the annotated frame
         cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
         if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
# Release the video capture object and close the display window
print(poselist)
cap.release()
writer.release()
cv2.destroyAllWindows()


