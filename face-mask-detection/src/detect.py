
import cv2

def detect_mask():

    print("Mask detection module initialized.")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.putText(frame,"Face Mask Detection Demo",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Mask Detection",frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
