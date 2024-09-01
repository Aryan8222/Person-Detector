import cv2
import torch
import numpy as np
from yolov8 import YOLOv8
from deep_sort_pytorch.deep_sort import DeepSort
import argparse

model = YOLOv8('yolov8x.pt')  


deepsort = DeepSort()

def main(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
   
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

       
        results = model(frame)
        bboxes, confs, classes = [], [], []

     
        for i, det in enumerate(results.xyxy[0]):
            if int(det[-1]) == 0:  # Person class
                x1, y1, x2, y2, conf, cls = det
                bboxes.append([x1, y1, x2, y2])
                confs.append(conf.item())
                classes.append(int(cls))

        if len(bboxes) > 0:
            bboxes = np.array(bboxes)
            confs = np.array(confs)
            outputs = deepsort.update(bboxes, confs, classes, frame)

          
            for output in outputs:
                x1, y1, x2, y2, track_id, class_id = output
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for people
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

      
        out.write(frame)

     
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, help="Path to input video")
    parser.add_argument("--output_path", type=str, help="Path to output video")
    args = parser.parse_args()

    main(args.video_path, args.output_path)
