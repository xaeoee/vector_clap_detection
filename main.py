import cv2
import mediapipe as mp

# MediaPipe Hand 모델 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 실시간 카메라 열기
cap = cv2.VideoCapture(0)  # 0번 카메라 사용

with mp_hands.Hands(
    static_image_mode=False,  # False로 설정하면 동영상에서도 작동
    max_num_hands=2,  # 감지할 손의 최대 개수
    min_detection_confidence=0.3,  # 감지 임계값
    min_tracking_confidence=0.3  # 추적 임계값
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오 읽기가 완료되었습니다.")
            break

        # get the height and width of the image. _ is number of channel but not going to be used.
        h, w, _ = frame.shape
        # print(h, w)
        # BGR을 RGB로 변환 (MediaPipe는 RGB 입력 필요)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 성능을 위해 이미지를 수정 불가능하게 설정
        frame_rgb.flags.writeable = False

        # 손 인식 수행
        results = hands.process(frame_rgb)

        # 다시 이미지를 수정 가능하게 설정
        frame_rgb.flags.writeable = True

        # 손 인식 결과 시각화
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 손목 랜드마크 좌표 출력
                wrist_landmark = hand_landmarks.landmark[0]
                five_landmark = hand_landmarks.landmark[5]
                st_landmark = hand_landmarks.landmark[17]
                wrist = (int(wrist_landmark.x * w), int(wrist_landmark.y * h))
                five = (int(five_landmark.x * w), int(five_landmark.y * h))
                st = (int(st_landmark.x * w), int(st_landmark.y * h))
                # print(f"Wrist landmark: x={wrist_landmark.x}, y={wrist_landmark.y}, z={wrist_landmark.z}")
                # print((int(wrist_landmark.x * w), int(wrist_landmark.y * h)))
                cv2.line(frame, wrist, st, color=(0, 255, 0), thickness=2)
                cv2.line(frame, wrist, five, color=(0, 255, 0), thickness=2)
                print(wrist, st)
                # print(f"one landmark: x={one_landmark.x}, y={one_landmark.y}, z={one_landmark.z}")
                # print(f"st landmark: x={st_landmark.x}, y={st_landmark.y}, z={st_landmark.z}")

        # 결과 프레임 출력
        cv2.imshow("Hand Detection", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
