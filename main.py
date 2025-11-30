import os
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# ---------------------------
# STATIC IMAGE HAND PROCESSING
# ---------------------------

IMAGE_FILES = []

def move_hands():
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:

        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.flip(cv2.imread(file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            print("Handedness:", results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue

            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            for hand_landmarks in results.multi_hand_landmarks:
                print("hand_landmarks:", hand_landmarks)

                print(
                    "Index finger tip coordinates:",
                    hand_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ].x * image_width,
                    hand_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ].y * image_height,
                )

                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                cv2.imwrite(
                    f"/tmp/annotated_image{idx}.png",
                    cv2.flip(annotated_image, 1)
                )

            # Draw world landmarks if present
            if results.multi_hand_world_landmarks:
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    mp_drawing.plot_landmarks(
                        hand_world_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        azimuth=5
                    )


# ---------------------------
# WEBCAM LIVE HAND TRACKING + OPTIONAL MP4 SAVE
# ---------------------------

def get_input_capture(output_path=None):
    cap = cv2.VideoCapture(0)

    # Setup optional saving
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving webcam recording to: {output_path}")

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Process frame
            image.flags.writeable = False
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            image.flags.writeable = True
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Save if writer exists
            if out is not None:
                out.write(image)

            # Display
            cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
                break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


# ---------------------------
# MAIN EXECUTION
# ---------------------------

def main():
    move_hands()
    os.makedirs("outputs",exist_ok=True)
    get_input_capture(output_path="./outputs/hand_tracking.mp4")  # or None

if __name__ == "__main__":
    main()