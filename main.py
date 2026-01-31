import cv2

# Load cascades
face_front = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_profile = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
eyes_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    #print(f"Frame dimensions: {w}x{h}")

    # --- Frontal faces ---
    faces_front = face_front.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(40, 40)
    )

    for (x, y, fw, fh) in faces_front:
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        cv2.putText(
            frame, "Front",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # --- Left profile ---
    profiles = face_profile.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, pw, ph) in profiles:
        cv2.rectangle(frame, (x, y), (x+pw, y+ph), (255, 0, 0), 2)
        cv2.putText(
            frame, "Profile",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # --- Right profile
    gray_flipped = cv2.flip(gray, 1)
    profiles_flipped = face_profile.detectMultiScale(
        gray_flipped,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, pw, ph) in profiles_flipped:
        # Convert flipped coordinates back
        x = w - x - pw
        cv2.rectangle(frame, (x, y), (x+pw, y+ph), (255, 0, 0), 2)
        cv2.putText(
            frame, "Profile",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    eyes = eyes_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(20, 20)
    )

    for (x, y, ew, eh) in eyes:
        cv2.rectangle(frame, (x, y), (x+ew, y+eh), (0, 0, 255), 2)
        cv2.putText(
            frame, "Eye",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )
    
    if len(eyes) == 0:
        cv2.putText(
            frame, "Keep your eyes open!",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255), # color BGR
            2 # thickness
        )

    cv2.imshow("Frontal + Side Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    

cap.release()
cv2.destroyAllWindows()
