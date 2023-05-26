import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_angle2(image, joint_list):
    for joint in joint_list:
        a = np.array(joint[0])
        b = np.array(joint[1])
        c = np.array(joint[2])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        cv2.putText(image, str(angle),
                    tuple(np.multiply(b, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

    return image


cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        # define landmarks connections
        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            left_eye_inner = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y]
            right_eye_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]
            left_eye_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
            right_eye_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]

            angle_left_elbow = left_shoulder, left_elbow, left_wrist
            angle_right_elbow = right_shoulder, right_elbow, right_wrist
            angle_left_shoulder = left_hip, left_shoulder, left_wrist
            angle_right_shoulder = right_hip, right_shoulder, right_elbow
            angle_left_knee = left_hip, left_knee, left_ankle
            angle_right_knee = right_hip, right_knee, right_ankle
            angle_left_ankle = left_knee, left_ankle, left_foot_index
            angle_right_ankle = right_knee, right_ankle, right_foot_index
            angle_left_hip = left_shoulder, left_hip, left_knee
            angle_right_hip = right_shoulder, right_hip, right_knee
            angle_head = left_eye, nose, left_shoulder

            left_eye_midpoint = np.mean([left_eye_inner, left_eye_outer], axis=0)
            right_eye_midpoint = np.mean([right_eye_inner, right_eye_outer], axis=0)
            eye_midpoint = np.mean([left_eye_midpoint, right_eye_midpoint], axis=0)

            joint_list = angle_left_elbow, angle_right_elbow, angle_left_shoulder, angle_right_shoulder, angle_left_knee, angle_right_knee, angle_left_ankle, angle_right_ankle, angle_left_hip, angle_right_hip, angle_head

            calculate_angle2(image, joint_list)

            angle = np.arctan2(eye_midpoint[1] - nose[1], eye_midpoint[0] - nose[0]) * 180 / np.pi
            angle = np.abs(angle)

            cv2.putText(image, str(angle),
                        tuple(np.multiply(nose, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            print(landmarks)

        except:
            pass

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=2))

        cv2.imshow('Model Detections', image)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# başaramadık abi asdasdasda asdasdasd
