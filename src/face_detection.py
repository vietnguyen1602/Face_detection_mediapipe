import mediapipe as mp
import cv2

###
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
###
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image_RGB = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = face_detection.process(image_RGB)
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, bbox, (255, 0, 255), 2)
                # mp_drawing.draw_detection(image, detection)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()