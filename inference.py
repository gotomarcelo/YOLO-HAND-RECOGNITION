from ultralytics import YOLO
import cv2

model_path = 'C:\\Users\\u11886\\Documents\\PDI_teste\\PDI_teste\\runs\\pose\\train5\\weights\\last.pt'

video_path = './samples/TF1-contraste.mp4'
cap = cv2.VideoCapture(video_path)

model = YOLO(model_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]

    for result in results:
        for keypoint_indx, keypoint in enumerate(result.keypoints.xy):
            for coord in keypoint:
                x, y = int(coord[0]), int(coord[1])  # Convert to integers if needed
                cv2.putText(frame, str(keypoint_indx), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('img', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
