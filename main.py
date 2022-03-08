import cv2
from ultils.FaceDetectionModule import *

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (960, 640))
        img, bboxs = detector.findFaces(img)
        cTime = time.time()
        # tính FPS
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Image",  img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()