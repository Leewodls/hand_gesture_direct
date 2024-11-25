
# # # 직접 추출 ################# 이게 최종 학습한 추출 코드
# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import time, os

# # actions = ['meet', 'nice', 'hello', 'you', 'name', 'what']
# # seq_length = 30
# # secs_for_action = 30

# # # MediaPipe hands model
# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # hands = mp_hands.Hands(
# #     max_num_hands=2,
# #     min_detection_confidence=0.5,
# #     min_tracking_confidence=0.5)

# # cap = cv2.VideoCapture(1)

# # created_time = int(time.time())
# # os.makedirs('dataset', exist_ok=True)

# # while cap.isOpened():
# #     for idx, action in enumerate(actions):
# #         data = []

# #         ret, img = cap.read()

# #         img = cv2.flip(img, 1)

# #         cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
# #         cv2.imshow('img', img)
# #         cv2.waitKey(3000)

# #         start_time = time.time()

# #         while time.time() - start_time < secs_for_action:
# #             ret, img = cap.read()

# #             img = cv2.flip(img, 1)
# #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #             result = hands.process(img)
# #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# #             if result.multi_hand_landmarks is not None:
# #                 for res in result.multi_hand_landmarks:
# #                     joint = np.zeros((21, 4))
# #                     for j, lm in enumerate(res.landmark):
# #                         joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# #                     # Compute angles between joints
# #                     v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
# #                     v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
# #                     v = v2 - v1 # [20, 3]
# #                     # Normalize v
# #                     v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# #                     # Get angle using arcos of dot product
# #                     angle = np.arccos(np.einsum('nt,nt->n',
# #                         v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
# #                         v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

# #                     angle = np.degrees(angle) # Convert radian to degree

# #                     angle_label = np.array([angle], dtype=np.float32)
# #                     angle_label = np.append(angle_label, idx)

# #                     d = np.concatenate([joint.flatten(), angle_label])

# #                     data.append(d)

# #                     mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

# #             cv2.imshow('img', img)
# #             if cv2.waitKey(1) == ord('q'):
# #                 break

# #         data = np.array(data)
# #         print(action, data.shape)
# #         np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

# #         # Create sequence data
# #         full_seq_data = []
# #         for seq in range(len(data) - seq_length):
# #             full_seq_data.append(data[seq:seq + seq_length])

# #         full_seq_data = np.array(full_seq_data)
# #         print(action, full_seq_data.shape)
# #         np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
# #     break


# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import time
# # import os

# # # 인식할 손동작 리스트
# # actions = ['meet', 'nice', 'hello']
# # seq_length = 30
# # secs_for_action = 30

# # # MediaPipe Hands 초기화
# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # hands = mp_hands.Hands(
# #     max_num_hands=2,  # 최대 인식할 손의 개수
# #     min_detection_confidence=0.5,  # 최소 검출 신뢰도
# #     min_tracking_confidence=0.5  # 최소 추적 신뢰도
# # )

# # # 웹캠 열기
# # cap = cv2.VideoCapture(1)

# # created_time = int(time.time())
# # os.makedirs('dataset', exist_ok=True)

# # while cap.isOpened():
# #     for idx, action in enumerate(actions):
# #         data_both_hands = []

# #         ret, img = cap.read()

# #         img = cv2.flip(img, 1)

# #         cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
# #         cv2.imshow('img', img)
# #         cv2.waitKey(3000)

# #         start_time = time.time()

# #         while time.time() - start_time < secs_for_action:
# #             ret, img = cap.read()

# #             img = cv2.flip(img, 1)
# #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #             result = hands.process(img)
# #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# #             if result.multi_hand_landmarks is not None:
# #                 for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# #                     # 손의 종류 확인 (오른손, 왼손)
# #                     hand_label = hand_info.classification[0].label
# #                     joint = np.zeros((21, 4))
# #                     for j, lm in enumerate(hand_landmarks.landmark):
# #                         joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# #                     # 오른손과 왼손에 따라 좌표를 저장
# #                     coords = joint[:, :3].flatten()
# #                     # 라벨 추가
# #                     coords_with_label = np.append(coords, idx)
# #                     # 손의 종류에 따라 데이터 배열에 추가
# #                     if hand_label == 'Right':
# #                         data_both_hands.append(coords_with_label)
# #                     else:
# #                         data_both_hands.append(coords_with_label)

# #                     mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# #             cv2.imshow('img', img)
# #             if cv2.waitKey(1) == ord('q'):
# #                 break

# #         data_both_hands = np.array(data_both_hands)
# #         print(action, "Both hands:", data_both_hands.shape)
# #         np.save(os.path.join('dataset', f'raw_{action}_both_hands_{created_time}'), data_both_hands)

# #         # Create sequence data for both hands
# #         full_seq_data_both_hands = []
# #         for seq in range(len(data_both_hands) - seq_length):
# #             full_seq_data_both_hands.append(data_both_hands[seq:seq + seq_length])

# #         full_seq_data_both_hands = np.array(full_seq_data_both_hands)
# #         print(action, "Both hands sequence:", full_seq_data_both_hands.shape)
# #         np.save(os.path.join('dataset', f'seq_{action}_both_hands_{created_time}'), full_seq_data_both_hands)

# #     break

# # # 웹캠 해제
# # cap.release()



# # # ### 양 손 인식 수정해야하는 코드1
# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import time
# # import os

# # # 인식할 손동작 리스트
# # actions = ['meet', 'nice', 'hello']  # 인식할 손동작 리스트
# # seq_length = 30  # 시퀀스 길이
# # secs_for_action = 30  # 동작 수집 시간(초)

# # # MediaPipe Hands 초기화
# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # hands = mp_hands.Hands(
# #     max_num_hands=2,  # 최대 인식할 손의 개수
# #     min_detection_confidence=0.5,  # 최소 검출 신뢰도
# #     min_tracking_confidence=0.5  # 최소 추적 신뢰도
# # )

# # # 웹캠 열기
# # cap = cv2.VideoCapture(1)

# # created_time = int(time.time())
# # os.makedirs('dataset', exist_ok=True)  # 데이터셋 저장 디렉토리 생성

# # while cap.isOpened():
# #     for idx, action in enumerate(actions):
# #         data_right_hand = []  # 오른손 데이터를 저장할 리스트 초기화
# #         data_left_hand = []  # 왼손 데이터를 저장할 리스트 초기화

# #         ret, img = cap.read()  # 웹캠에서 이미지 읽기

# #         img = cv2.flip(img, 1)  # 좌우 반전

# #         # 동작 수집 대기 메시지 표시
# #         cv2.putText(img, f'동작 "{action.upper()}" 수집 대기 중...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
# #         cv2.imshow('img', img)
# #         cv2.waitKey(3000)  # 대기 시간 설정 (3초)

# #         start_time = time.time()  # 시작 시간 기록

# #         while time.time() - start_time < secs_for_action:
# #             ret, img = cap.read()  # 웹캠에서 이미지 읽기

# #             img = cv2.flip(img, 1)  # 좌우 반전
# #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
# #             result = hands.process(img)  # 손동작 인식 처리
# #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환

# #             if result.multi_hand_landmarks is not None:
# #                 for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# #                     # 손의 종류 확인 (오른손, 왼손)
# #                     hand_label = hand_info.classification[0].label
# #                     joint = np.zeros((21, 4))  # 관절 좌표를 저장할 배열 초기화
# #                     for j, lm in enumerate(hand_landmarks.landmark):
# #                         joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# #                     # 관절 좌표만 추출하여 데이터에 추가
# #                     coords = joint[:, :3].flatten()
# #                     coords_with_label = np.append(coords, idx)

# #                     if hand_label == 'Right':
# #                         data_right_hand.append(coords_with_label)
# #                     else:
# #                         data_left_hand.append(coords_with_label)

# #                     mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# #             cv2.imshow('img', img)
# #             if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 종료
# #                 break

# #         # 오른손과 왼손 데이터를 각각 NumPy 배열로 변환
# #         data_right_hand = np.array(data_right_hand)
# #         data_left_hand = np.array(data_left_hand)

# #         # 오른손과 왼손 데이터를 합침
# #         data_both_hands = np.concatenate((data_right_hand, data_left_hand), axis=0)

# #         print(action, "Both hands:", data_both_hands.shape)
# #         np.save(os.path.join('dataset', f'raw_{action}_both_hands_{created_time}'), data_both_hands)  # 데이터 저장

# #         # 양손 데이터를 시퀀스로 변환
# #         full_seq_data_both_hands = []
# #         for seq in range(len(data_both_hands) - seq_length):
# #             full_seq_data_both_hands.append(data_both_hands[seq:seq + seq_length])

# #         full_seq_data_both_hands = np.array(full_seq_data_both_hands)  # 양손 시퀀스 데이터를 NumPy 배열로 변환
# #         print(action, "Both hands sequence:", full_seq_data_both_hands.shape)
# #         np.save(os.path.join('dataset', f'seq_{action}_both_hands_{created_time}'), full_seq_data_both_hands)  # 데이터 저장

# #     break  # 한 번만 실행

# # # 웹캠 해제
# # cap.release()


# # # ### 양 손 인식 수정해야하는 코드2 /양 손 따로 저장
# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import time
# # import os

# # # 인식할 손동작 리스트
# # actions = ['meet', 'nice', 'hello']  # 인식할 손동작 리스트
# # seq_length = 30  # 시퀀스 길이
# # secs_for_action = 30  # 동작 수집 시간(초)

# # # MediaPipe Hands 초기화
# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # hands = mp_hands.Hands(
# #     max_num_hands=2,  # 최대 인식할 손의 개수
# #     min_detection_confidence=0.5,  # 최소 검출 신뢰도
# #     min_tracking_confidence=0.5  # 최소 추적 신뢰도
# # )

# # # 웹캠 열기
# # cap = cv2.VideoCapture(1)

# # created_time = int(time.time())
# # os.makedirs('dataset', exist_ok=True)  # 데이터셋 저장 디렉토리 생성

# # while cap.isOpened():
# #     for idx, action in enumerate(actions):
# #         data_right_hand = []  # 오른손 데이터를 저장할 리스트 초기화
# #         data_left_hand = []  # 왼손 데이터를 저장할 리스트 초기화

# #         ret, img = cap.read()  # 웹캠에서 이미지 읽기

# #         img = cv2.flip(img, 1)  # 좌우 반전

# #         # 동작 수집 대기 메시지 표시
# #         cv2.putText(img, f'동작 "{action.upper()}" 수집 대기 중...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
# #         cv2.imshow('img', img)
# #         cv2.waitKey(2000)  # 대기 시간 설정 (3초)

# #         start_time = time.time()  # 시작 시간 기록

# #         while time.time() - start_time < secs_for_action:
# #             ret, img = cap.read()  # 웹캠에서 이미지 읽기

# #             img = cv2.flip(img, 1)  # 좌우 반전
# #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
# #             result = hands.process(img)  # 손동작 인식 처리
# #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환

# #             if result.multi_hand_landmarks is not None:
# #                 for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# #                     # 손의 종류 확인 (오른손, 왼손)
# #                     hand_label = hand_info.classification[0].label
# #                     joint = np.zeros((21, 4))  # 관절 좌표를 저장할 배열 초기화
# #                     for j, lm in enumerate(hand_landmarks.landmark):
# #                         joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# #                     # 관절 좌표만 추출하여 데이터에 추가
# #                     coords = joint[:, :3].flatten()
# #                     coords_with_label = np.append(coords, idx)  # 손동작 레이블 추가

# #                     if hand_label == 'Right':
# #                         data_right_hand.append(coords_with_label)
# #                     else:
# #                         data_left_hand.append(coords_with_label)

# #                     mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# #             cv2.imshow('img', img)
# #             if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 종료
# #                 break

# #         # 오른손과 왼손 데이터를 각각 NumPy 배열로 변환
# #         data_right_hand = np.array(data_right_hand)
# #         data_left_hand = np.array(data_left_hand)

# #         # 오른손과 왼손 데이터에 손 레이블 추가하여 저장
# #         labeled_data_right_hand = np.hstack((data_right_hand, np.zeros((data_right_hand.shape[0], 1))))  # 오른손 레이블 추가
# #         labeled_data_left_hand = np.hstack((data_left_hand, np.ones((data_left_hand.shape[0], 1))))  # 왼손 레이블 추가

# #         labeled_data_both_hands = np.concatenate((labeled_data_right_hand, labeled_data_left_hand), axis=0)

# #         # 손동작 레이블 추가하여 저장
# #         labeled_data_with_action = np.hstack((labeled_data_both_hands, np.full((labeled_data_both_hands.shape[0], 1), idx)))

# #         print(action, "Both hands:", labeled_data_with_action.shape)
# #         np.save(os.path.join('dataset', f'raw_{action}_labeled_both_hands_{created_time}'), labeled_data_with_action)  # 레이블이 추가된 데이터 저장
        
# #         # Create sequence data using left and right hand data
# #         # 왼손과 오른손 데이터를 시퀀스로 생성
# #         sequence_data = []
# #         for seq in range(len(labeled_data_with_action) - seq_length):
# #             sequence_data.append(labeled_data_with_action[seq:seq + seq_length])

# #         sequence_data = np.array(sequence_data)
# #         print(action, "Sequence data shape:", sequence_data.shape)
# #         np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), sequence_data)
        
# #     break  # 한 번만 실행

# # # 웹캠 해제
# # cap.release()

#위 코드 수정본
import cv2
import mediapipe as mp
import numpy as np
import time
import os

# 인식할 손동작 리스트
actions = ['thanks', 'sorry']
seq_length = 30
secs_for_action = 4
num_repeats = 15  # 각 동작마다 30개의 시퀀스를 수집

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # 최대 2개의 손을 인식
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 열기
cap = cv2.VideoCapture(1)

# 웹캠이 제대로 열리지 않는 경우 다른 인덱스를 시도
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

for idx, action in enumerate(actions):
    for repeat in range(num_repeats):
        data_right_hand = []
        data_left_hand = []

        ret, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.putText(img, f'Action "{action.upper()}" ({repeat+1}/{num_repeats}) Wait...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(2000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                    hand_label = hand_info.classification[0].label
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(hand_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                    v = v2 - v1
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # 내적 값 클리핑
                    dot_product = np.einsum('nt,nt->n', v, v)
                    dot_product = np.clip(dot_product, -1.0, 1.0)

                    angle = np.arccos(dot_product)
                    angle = np.degrees(angle)

                    d = np.concatenate([joint.flatten(), angle])

                    if hand_label == 'Right':
                        data_right_hand.append(d)
                    else:
                        data_left_hand.append(d)

                # 양손의 랜드마크 그리기
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        data_right_hand = np.array(data_right_hand)
        data_left_hand = np.array(data_left_hand)

        np.save(os.path.join('dataset', f'seq_{action}_{created_time}_{repeat}_right.npy'), data_right_hand)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}_{repeat}_left.npy'), data_left_hand)

# 모든 동작 수집 완료 후 종료
cap.release()
cv2.destroyAllWindows()
print("모든 동작에 대한 데이터 수집이 완료되었습니다.")

