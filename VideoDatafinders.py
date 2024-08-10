from ultralytics import YOLO
import numpy as np
from numba import jit ,cuda
target_backend=cuda
model=YOLO('yolov8m-pose.pt')
list=[]
B=0
n=0
results=model(source="WIN_20240809_17_35_03_Pro.mp4",show=True,conf=0.3)
for r in results:
        keypoints=r.keypoints
        B=np.array(keypoints.data)
        print(B.shape)
        if B.shape[1]==0:
                break
        A=B[0].reshape(51)
        list.append(np.array(A))
        n=n+1
print(n)
list=np.array(list)
print(list.shape)
np.save("handclaps.npy" ,list)