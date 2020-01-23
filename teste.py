import cv2
cap = cv2.VideoCapture("rtsp://before:beforeti@192.168.1.196:554/live/ch0")
cap.set(3,1280)
cap.set(4,720)

video_writer = cv2.VideoWriter('recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, (1280,720))
while cap.isOpened():
    isSuccess,frame = cap.read()
    if isSuccess:            
        cv2.imshow('face Capture', frame)
        video_writer.write(frame)
        
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    print("Takepic")
cap.release()