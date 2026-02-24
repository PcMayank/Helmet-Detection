import cv2
import time
import threading
from ultralytics import YOLO


# Path to trained model
MODEL_PATH = "../model/best.pt"


# Beep sound (Windows)
def play_beep():
    try:
        import winsound
        winsound.Beep(1000, 500)
    except:
        pass


def main():

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera error")
        return

    print("Helmet Detection Started (Press Q to quit)")

    last_beep = 0
    beep_gap = 3  # seconds

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))

        # LOW confidence for demo
        results = model.predict(frame, conf=0.2, verbose=False)

        annotated = results[0].plot()

        helmet_score = 0
        nohelmet_score = 0

        # Read detections
        for box in results[0].boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            if label == "with_helmet":
                helmet_score = max(helmet_score, conf)

            if label == "without_helmet":
                nohelmet_score = max(nohelmet_score, conf)

        # Decision (simple & stable)
        if helmet_score > nohelmet_score and helmet_score > 0.3:

            status = "Helmet Detected ✅"
            color = (0, 255, 0)

        elif nohelmet_score > helmet_score and nohelmet_score > 0.3:

            status = "No Helmet ❌"
            color = (0, 0, 255)

            # Beep
            now = time.time()
            if now - last_beep > beep_gap:
                last_beep = now
                threading.Thread(
                    target=play_beep,
                    daemon=True
                ).start()

        else:

            status = "No Clear Detection"
            color = (0, 255, 255)

        # Show confidence
        info = f"H:{helmet_score:.2f}  NH:{nohelmet_score:.2f}"

        # Draw text
        cv2.putText(
            annotated,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

        cv2.putText(
            annotated,
            info,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Helmet Safety System (Demo Mode)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()