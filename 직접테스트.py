# # # # # ##### 마지막 테스트 해봤던 코드##
# # # # # import cv2
# # # # # import mediapipe as mp
# # # # # import numpy as np
# # # # # from tensorflow.keras.models import load_model

# # # # # # 인식할 손동작 리스트
# # # # # actions = ['meet', 'nice', 'hello', 'you', 'name', 'what']
# # # # # seq_length = 30  # 시퀀스 길이

# # # # # # 훈련된 모델 로드
# # # # # model = load_model('models/n_model.keras')

# # # # # # MediaPipe Hands 초기화
# # # # # mp_hands = mp.solutions.hands
# # # # # mp_drawing = mp.solutions.drawing_utils
# # # # # hands = mp_hands.Hands(
# # # # #     max_num_hands=2,  # 최대 인식할 손의 개수
# # # # #     min_detection_confidence=0.5,  # 최소 검출 신뢰도
# # # # #     min_tracking_confidence=0.5  # 최소 추적 신뢰도
# # # # # )

# # # # # # 웹캠 열기
# # # # # cap = cv2.VideoCapture(1)

# # # # # seq_right = []  # 오른손 랜드마크 시퀀스
# # # # # seq_left = []   # 왼손 랜드마크 시퀀스
# # # # # prev_action = None  # 이전 동작
# # # # # output_file = 'captions.txt'  # 출력 파일명

# # # # # # 파일을 쓰기 모드로 열기
# # # # # with open(output_file, 'w', encoding='utf-8') as file:
# # # # #     while cap.isOpened():
# # # # #         ret, img = cap.read()
# # # # #         if not ret:
# # # # #             break

# # # # #         img0 = img.copy()
# # # # #         img = cv2.flip(img, 1)  # 좌우 반전
# # # # #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
# # # # #         result = hands.process(img)  # 손동작 인식 처리
# # # # #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환

# # # # #         if result.multi_hand_landmarks is not None:
# # # # #             for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# # # # #                 # 손의 종류 확인 (오른손, 왼손)
# # # # #                 hand_label = hand_info.classification[0].label
# # # # #                 joint = np.zeros((21, 4))
# # # # #                 for j, lm in enumerate(hand_landmarks.landmark):
# # # # #                     joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # # # #                 # 관절 간의 각도 계산
# # # # #                 v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # # # #                 v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # # # #                 v = v2 - v1
# # # # #                 v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # # # #                 # 각도의 코사인값을 이용하여 각도 계산
# # # # #                 angle = np.arccos(np.einsum('nt,nt->n',
# # # # #                                             v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
# # # # #                                             v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
# # # # #                 angle = np.degrees(angle)  # 라디안에서 도 단위로 변환

# # # # #                 d = np.concatenate([joint.flatten(), angle])
                
# # # # #                 # 오른손과 왼손에 따라 시퀀스를 저장
# # # # #                 if hand_label == 'Right':
# # # # #                     seq_right.append(d)
# # # # #                 else:
# # # # #                     seq_left.append(d)

# # # # #                 # 손동작 랜드마크 그리기
# # # # #                 mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # # # #                 # 양손의 시퀀스가 시퀀스 길이에 도달하면 모델에 입력하고 결과 예측
# # # # #                 if len(seq_right) >= seq_length and len(seq_left) >= seq_length:
# # # # #                     input_data_right = np.expand_dims(np.array(seq_right[-seq_length:], dtype=np.float32), axis=0)
# # # # #                     input_data_left = np.expand_dims(np.array(seq_left[-seq_length:], dtype=np.float32), axis=0)

# # # # #                     y_pred_right = model.predict(input_data_right).squeeze()
# # # # #                     y_pred_left = model.predict(input_data_left).squeeze()

# # # # #                     # 양손의 예측 결과가 모두 일정 확률 이상인 경우에만 자막 출력
# # # # #                     if y_pred_right.max() >= 0.9 and y_pred_left.max() >= 0.9:
# # # # #                         i_pred_right = int(np.argmax(y_pred_right))
# # # # #                         i_pred_left = int(np.argmax(y_pred_left))
# # # # #                         action = actions[i_pred_right]
                        
                            
# # # # #                             # 동작이 변경되었을 때만 파일에 기록
# # # # #                         if prev_action != action:
# # # # #                             cv2.putText(img, f'{action.upper()}', org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
# # # # #                                     fontScale=1, color=(255, 255, 255), thickness=2)
# # # # #                             file.write(f"{action}\n")
# # # # #                             file.flush()  # 버퍼에 있는 내용을 즉시 파일에 기록
# # # # #                             prev_action = action

# # # # #         cv2.imshow('img', img)  # 화면에 이미지 출력
# # # # #         if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 종료
# # # # #             break

# # # # # cap.release()  # 웹캠 해제
# # # # # cv2.destroyAllWindows()  # 모든 창 닫기


     
# # # # # ### 새 파일 생성하는 코드
# # # # # import cv2
# # # # # import mediapipe as mp
# # # # # import numpy as np
# # # # # from tensorflow.keras.models import load_model

# # # # # # 인식할 손동작 리스트
# # # # # actions = ['meet', 'nice', 'hello']
# # # # # seq_length = 30  # 시퀀스 길이

# # # # # # 훈련된 모델 로드
# # # # # model = load_model('models/model_b.keras')

# # # # # # MediaPipe Hands 초기화
# # # # # mp_hands = mp.solutions.hands
# # # # # mp_drawing = mp.solutions.drawing_utils
# # # # # hands = mp_hands.Hands(
# # # # #     max_num_hands=2,  # 최대 인식할 손의 개수
# # # # #     min_detection_confidence=0.5,  # 최소 검출 신뢰도
# # # # #     min_tracking_confidence=0.5  # 최소 추적 신뢰도
# # # # # )

# # # # # # 웹캠 열기
# # # # # cap = cv2.VideoCapture(1)

# # # # # seq_right = []  # 오른손 랜드마크 시퀀스
# # # # # seq_left = []   # 왼손 랜드마크 시퀀스
# # # # # prev_action = None  # 이전 동작을 저장할 변수
# # # # # output_file = 'captions.txt'  # 출력 파일명
# # # # # file_index = 1

# # # # # # 파일을 쓰기 모드로 열기
# # # # # def create_new_file():
# # # # #     global output_file, file_index
# # # # #     output_file = f'captions{file_index}.txt'
# # # # #     file_index += 1
# # # # #     return open(output_file, 'w', encoding='utf-8')

# # # # # file = create_new_file()

# # # # # while cap.isOpened():
# # # # #     ret, img = cap.read()
# # # # #     if not ret:
# # # # #         break

# # # # #     img0 = img.copy()
# # # # #     img = cv2.flip(img, 1)  # 좌우 반전
# # # # #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
# # # # #     result = hands.process(img)  # 손동작 인식 처리
# # # # #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환

# # # # #     if result.multi_hand_landmarks is not None:
# # # # #         for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# # # # #             # 손의 종류 확인 (오른손, 왼손)
# # # # #             hand_label = hand_info.classification[0].label
# # # # #             joint = np.zeros((21, 4))
# # # # #             for j, lm in enumerate(hand_landmarks.landmark):
# # # # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # # # #             # 관절 간의 각도 계산
# # # # #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # # # #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # # # #             v = v2 - v1
# # # # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # # # #             # 각도의 코사인값을 이용하여 각도 계산
# # # # #             angle = np.arccos(np.einsum('nt,nt->n',
# # # # #                                         v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
# # # # #                                         v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
# # # # #             angle = np.degrees(angle)  # 라디안에서 도 단위로 변환

# # # # #             d = np.concatenate([joint.flatten(), angle])
            
# # # # #             # 오른손과 왼손에 따라 시퀀스를 저장
# # # # #             if hand_label == 'Right':
# # # # #                 seq_right.append(d)
# # # # #             else:
# # # # #                 seq_left.append(d)

# # # # #             # 손동작 랜드마크 그리기
# # # # #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # # # #             # 양손의 시퀀스가 시퀀스 길이에 도달하면 모델에 입력하고 결과 예측
# # # # #             if len(seq_right) >= seq_length and len(seq_left) >= seq_length:
# # # # #                 input_data_right = np.expand_dims(np.array(seq_right[-seq_length:], dtype=np.float32), axis=0)
# # # # #                 input_data_left = np.expand_dims(np.array(seq_left[-seq_length:], dtype=np.float32), axis=0)

# # # # #                 y_pred_right = model.predict(input_data_right).squeeze()
# # # # #                 y_pred_left = model.predict(input_data_left).squeeze()

# # # # #                 # 양손의 예측 결과가 일정 확률 이상이면 자막 출력
# # # # #                 if y_pred_right.max() >= 0.9 and y_pred_left.max() >= 0.9:
# # # # #                     right_action_index = np.argmax(y_pred_right)
# # # # #                     left_action_index = np.argmax(y_pred_left)
                    
# # # # #                     # 각 손의 예측 결과가 모델의 예측과 유사한지 확인
# # # # #                     if y_pred_right[right_action_index] >= 0.9 and y_pred_left[left_action_index] >= 0.9:
# # # # #                         action = actions[right_action_index]

# # # # #                         # 동작이 변경되었을 때만 자막 출력
# # # # #                         if prev_action != action:
# # # # #                             cv2.putText(img, f'{action.upper()}', org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
# # # # #                                         fontScale=1, color=(255, 255, 255), thickness=2)
# # # # #                             file.write(f"{action}\n")
# # # # #                             file.flush()  # 버퍼에 있는 내용을 즉시 파일에 기록
# # # # #                             prev_action = action

# # # # #     cv2.imshow('img', img)  # 화면에 이미지 출력
# # # # #     key = cv2.waitKey(1)
# # # # #     if key == ord('q'):  # 'q'
# # # # #         break
# # # # #     elif key == ord('n'):
# # # # #         file.close()  # 파일을 닫음
# # # # #         file = create_new_file()  # 새 파일을 생성하여 염

# # # # # # 웹캠 해제 및 창 닫기
# # # # # cap.release()
# # # # # cv2.destroyAllWindows()



# # # # # ### 양 손 인식 수정해야하는 코드
# # # # # import cv2
# # # # # import mediapipe as mp
# # # # # import numpy as np
# # # # # from tensorflow.keras.models import load_model

# # # # # # 인식할 손동작 리스트
# # # # # actions = ['meet', 'nice', 'hello']
# # # # # seq_length = 99  # 시퀀스 길이

# # # # # # 훈련된 모델 로드
# # # # # model = load_model('models/model_c.keras')

# # # # # # MediaPipe Hands 초기화
# # # # # mp_hands = mp.solutions.hands
# # # # # mp_drawing = mp.solutions.drawing_utils
# # # # # hands = mp_hands.Hands(
# # # # #     max_num_hands=2,  # 최대 인식할 손의 개수
# # # # #     min_detection_confidence=0.5,  # 최소 검출 신뢰도
# # # # #     min_tracking_confidence=0.5  # 최소 추적 신뢰도
# # # # # )

# # # # # # 웹캠 열기
# # # # # cap = cv2.VideoCapture(1)

# # # # # seq_both_hands = []  # 오른손과 왼손 랜드마크 및 각도 시퀀스
# # # # # prev_action = None  # 이전 동작을 저장할 변수
# # # # # output_file = 'captions.txt'  # 출력 파일명
# # # # # file_index = 1

# # # # # # 파일을 쓰기 모드로 열기
# # # # # def create_new_file():
# # # # #     global output_file, file_index
# # # # #     output_file = f'captions{file_index}.txt'
# # # # #     file_index += 1
# # # # #     return open(output_file, 'w', encoding='utf-8')

# # # # # file = create_new_file()

# # # # # while cap.isOpened():
# # # # #     ret, img = cap.read()
# # # # #     if not ret:
# # # # #         break

# # # # #     img0 = img.copy()
# # # # #     img = cv2.flip(img, 1)  # 좌우 반전
# # # # #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
# # # # #     result = hands.process(img)  # 손동작 인식 처리
# # # # #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환

# # # # #     if result.multi_hand_landmarks is not None:
# # # # #         for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# # # # #             # 손의 종류 확인 (오른손, 왼손)
# # # # #             hand_label = hand_info.classification[0].label
# # # # #             joint = np.zeros((21, 4))
# # # # #             for j, lm in enumerate(hand_landmarks.landmark):
# # # # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # # # #             # 관절 간의 각도 계산
# # # # #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # # # #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # # # #             v = v2 - v1
# # # # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # # # #             # 각도의 코사인값을 이용하여 각도 계산
# # # # #             angle = np.arccos(np.einsum('nt,nt->n',
# # # # #                                         v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
# # # # #                                         v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
# # # # #             angle = np.degrees(angle)  # 라디안에서 도 단위로 변환

# # # # #             d = np.concatenate([joint.flatten(), angle])
            
# # # # #             # 오른손과 왼손 데이터를 함께 저장
# # # # #             seq_both_hands.append(d)

# # # # #             # 손동작 랜드마크 그리기
# # # # #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # # # #             # 시퀀스 길이에 도달하면 모델에 입력하고 결과 예측
# # # # #             if len(seq_both_hands) >= seq_length:
# # # # #                 input_data = np.expand_dims(np.array(seq_both_hands[-seq_length:], dtype=np.float32), axis=0)

# # # # #                 y_pred = model.predict(input_data)  # 수정된 부분

# # # # #                 # 예측 결과가 일정 확률 이상이면 자막 출력
# # # # #                 if y_pred.max() >= 0.9:
# # # # #                     action_index = np.argmax(y_pred)
# # # # #                     action = actions[action_index]

# # # # #                     # 이전 동작이 없거나 이전 동작과 다른 경우에만 자막 출력
# # # # #                     if prev_action != action:
# # # # #                         cv2.putText(img, f'{action.upper()}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
# # # # #                                     fontScale=1, color=(255, 255, 255), thickness=2)
# # # # #                         file.write(f"{action}\n")
# # # # #                         file.flush()  # 버퍼에 있는 내용을 즉시 파일에 기록
# # # # #                         prev_action = action

# # # # #     cv2.imshow('img', img)  # 화면에 이미지 출력
# # # # #     key = cv2.waitKey(1)
# # # # #     if key == ord('q'):  # 'q'
# # # # #         break
# # # # #     elif key == ord('n'):
# # # # #         file.close()  # 파일을 닫음
# # # # #         file = create_new_file()  # 새 파일을 생성하여 염

# # # # # # 웹캠 해제 및 창 닫기
# # # # # cap.release()
# # # # # cv2.destroyAllWindows()


# # # # #######최초 코드
# # # # # import cv2
# # # # # import mediapipe as mp
# # # # # import numpy as np
# # # # # from tensorflow.keras.models import load_model
# # # # # import time

# # # # # actions = ['meet', 'nice', 'hello', 'you', 'name', 'what']
# # # # # seq_length = 30

# # # # # model = load_model('models/n1_model.keras')

# # # # # # MediaPipe hands model
# # # # # mp_hands = mp.solutions.hands
# # # # # mp_drawing = mp.solutions.drawing_utils
# # # # # hands = mp_hands.Hands(
# # # # #     max_num_hands=2,  # 최대 2개의 손 인식
# # # # #     min_detection_confidence=0.5,
# # # # #     min_tracking_confidence=0.5)

# # # # # cap = cv2.VideoCapture(1)

# # # # # seq = []  # 좌표 시퀀스
# # # # # prev_action = None

# # # # # while cap.isOpened():
# # # # #     ret, img = cap.read()
# # # # #     img0 = img.copy()

# # # # #     img = cv2.flip(img, 1)
# # # # #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # # # #     result = hands.process(img)
# # # # #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# # # # #     if result.multi_hand_landmarks is not None:
# # # # #         for res in result.multi_hand_landmarks:
# # # # #             joint = np.zeros((21, 4))
# # # # #             for j, lm in enumerate(res.landmark):
# # # # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # # # #             # Compute angles between joints
# # # # #             v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
# # # # #             v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
# # # # #             v = v2 - v1 # [20, 3]
# # # # #             # Normalize v
# # # # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # # # #             # Get angle using arcos of dot product
# # # # #             angle = np.arccos(np.einsum('nt,nt->n',
# # # # #                 v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
# # # # #                 v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

# # # # #             angle = np.degrees(angle) # Convert radian to degree

# # # # #             d = np.concatenate([joint.flatten(), angle])
# # # # #             seq.append(d)

# # # # #             mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

# # # # #             # 좌표 시퀀스가 시퀀스 길이에 도달하면 모델에 입력하고 결과 예측
# # # # #             if len(seq) >= seq_length:
# # # # #                 input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

# # # # #                 y_pred = model.predict(input_data).squeeze()

# # # # #                 # 이전 동작과 현재 동작이 다르거나, 현재 동작의 확률이 일정 이상인 경우에만 자막 출력
# # # # #                 if prev_action != actions[np.argmax(y_pred)] or y_pred.max() >= 0.9:
# # # # #                     i_pred = int(np.argmax(y_pred))
# # # # #                     action = actions[i_pred]
                        
# # # # #                     if prev_action != action:
# # # # #                         cv2.putText(img, f'{"The word "+action.upper()+"was recognized."}', org=(0, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
# # # # #                         cv2.imshow('img', img)  # 자막 표시
                        
# # # # #                         prev_action = action

# # # # #     cv2.imshow('img', img)
# # # # #     if cv2.waitKey(1) == ord('q'):
# # # # #         break

# # # # # cap.release()
# # # # # cv2.destroyAllWindows()

# # # # # ####수정 전 코드 /자막 업데이트 오류
# # # # # import cv2
# # # # # import mediapipe as mp
# # # # # import numpy as np
# # # # # import tensorflow as tf
# # # # # from collections import deque

# # # # # # MediaPipe 초기화
# # # # # mp_hands = mp.solutions.hands
# # # # # mp_drawing = mp.solutions.drawing_utils

# # # # # # TensorFlow 모델 로드
# # # # # model = tf.keras.models.load_model('models/jm_model.keras')

# # # # # # 라벨 (단어) 리스트
# # # # # labels = ['meet', 'nice', 'hello']  # 모델이 예측할 단어들

# # # # # # 웹캠 초기화
# # # # # cap = cv2.VideoCapture(1)

# # # # # # 손 좌표 추출 함수
# # # # # def extract_hand_landmarks(results):
# # # # #     landmarks = []
# # # # #     if results.multi_hand_landmarks:
# # # # #         for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
# # # # #             hand_landmarks_array = np.array(
# # # # #                 [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
# # # # #             ).flatten()
# # # # #             # 손의 종류를 나타내는 추가 특징 (오른손: 1, 왼손: 0)
# # # # #             hand_type = 1 if hand_info.classification[0].label == 'Right' else 0
# # # # #             hand_landmarks_array = np.append(hand_landmarks_array, hand_type)
# # # # #             landmarks.append(hand_landmarks_array)
# # # # #     return np.array(landmarks)

# # # # # # 입력 데이터를 시퀀스 길이에 맞게 재구성하는 함수
# # # # # def pad_landmarks(landmarks, target_length=30, num_features=65):
# # # # #     if landmarks.size == 0:
# # # # #         return np.zeros((target_length, num_features))
    
# # # # #     num_landmarks = landmarks.shape[0] // num_features
# # # # #     num_padding = max(0, target_length - num_landmarks)
# # # # #     padded_landmarks = np.zeros((target_length, num_features))
    
# # # # #     for i in range(min(num_landmarks, target_length)):
# # # # #         padded_landmarks[i, :num_features] = landmarks[i*num_features:(i+1)*num_features]
    
# # # # #     return padded_landmarks

# # # # # # 최근 예측된 단어를 저장하기 위한 데크 초기화
# # # # # recent_predictions = deque(maxlen=30)
# # # # # current_prediction = None
# # # # # display_label = ""  # 화면에 표시할 라벨

# # # # # with mp_hands.Hands(
# # # # #     static_image_mode=False,
# # # # #     max_num_hands=2,
# # # # #     min_detection_confidence=0.5,
# # # # #     min_tracking_confidence=0.5) as hands:

# # # # #     while cap.isOpened():
# # # # #         ret, frame = cap.read()
# # # # #         if not ret:
# # # # #             break

# # # # #         # BGR 이미지를 RGB로 변환
# # # # #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # # # #         image = cv2.flip(image, 1)  # 좌우 반전
# # # # #         image.flags.writeable = False

# # # # #         # MediaPipe로 손 인식
# # # # #         results = hands.process(image)

# # # # #         # BGR 이미지로 변환
# # # # #         image.flags.writeable = True
# # # # #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# # # # #         # 손 좌표 추출
# # # # #         landmarks = extract_hand_landmarks(results)

# # # # #         if landmarks.size != 0:
# # # # #             # 입력 데이터를 시퀀스 길이에 맞게 패딩
# # # # #             padded_landmarks = pad_landmarks(landmarks)
            
# # # # #             # 모델 예측을 위해 차원 추가
# # # # #             padded_landmarks = np.expand_dims(padded_landmarks, axis=0)

# # # # #             # 모델 예측
# # # # #             predictions = model.predict(padded_landmarks)
# # # # #             predicted_label = labels[np.argmax(predictions)]

# # # # #             # 최근 예측된 단어와 비교하여 중복된 동작을 방지
# # # # #             if predicted_label != current_prediction:
# # # # #                 current_prediction = predicted_label
# # # # #                 display_label = predicted_label  # 새로운 예측 단어로 업데이트
# # # # #                 print(display_label)  # 콘솔에 예측된 자막 출력

# # # # #         # MediaPipe 손 랜드마크 그리기
# # # # #         if results.multi_hand_landmarks:
# # # # #             for hand_landmarks in results.multi_hand_landmarks:
# # # # #                 mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # # # #         # 자막 출력
# # # # #         if display_label:
# # # # #             cv2.putText(image, display_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# # # # #         # 결과 이미지 표시
# # # # #         cv2.imshow('Hand Gesture Recognition', image)

# # # # #         if cv2.waitKey(10) & 0xFF == ord('q'):
# # # # #             break

# # # # # cap.release()
# # # # # cv2.destroyAllWindows()

# # # # # 수정 후 코드
# # # # # import cv2
# # # # # import mediapipe as mp
# # # # # import numpy as np
# # # # # import tensorflow as tf

# # # # # # 학습된 모델 로드
# # # # # model = tf.keras.models.load_model('models/hand_gesture_model.keras')

# # # # # # 인식할 손동작 리스트
# # # # # actions = ['만나다', 'nice', 'hello']
# # # # # seq_length = 30

# # # # # # MediaPipe Hands 초기화
# # # # # mp_hands = mp.solutions.hands
# # # # # mp_drawing = mp.solutions.drawing_utils
# # # # # hands = mp_hands.Hands(
# # # # #     max_num_hands=2,  # 최대 2개의 손을 인식
# # # # #     min_detection_confidence=0.5,
# # # # #     min_tracking_confidence=0.5
# # # # # )

# # # # # # 웹캠 열기
# # # # # cap = cv2.VideoCapture(1)

# # # # # # 웹캠이 제대로 열리지 않는 경우 다른 인덱스를 시도
# # # # # if not cap.isOpened():
# # # # #     cap = cv2.VideoCapture(0)
# # # # # if not cap.isOpened():
# # # # #     print("웹캠을 열 수 없습니다.")
# # # # #     exit()

# # # # # sequence = []
# # # # # action_seq = []

# # # # # while cap.isOpened():
# # # # #     ret, img = cap.read()
# # # # #     if not ret:
# # # # #         break

# # # # #     img = cv2.flip(img, 1)
# # # # #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # # # #     result = hands.process(img)
# # # # #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# # # # #     if result.multi_hand_landmarks is not None:
# # # # #         for hand_landmarks in result.multi_hand_landmarks:
# # # # #             joint = np.zeros((21, 4))
# # # # #             for j, lm in enumerate(hand_landmarks.landmark):
# # # # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # # # #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # # # #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # # # #             v = v2 - v1
# # # # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # # # #             # 내적 값 클리핑
# # # # #             dot_product = np.einsum('nt,nt->n', v, v)
# # # # #             dot_product = np.clip(dot_product, -1.0, 1.0)

# # # # #             angle = np.arccos(dot_product)
# # # # #             angle = np.degrees(angle)

# # # # #             d = np.concatenate([joint.flatten(), angle])

# # # # #             sequence.append(d)
# # # # #             if len(sequence) > seq_length:
# # # # #                 sequence.pop(0)

# # # # #             if len(sequence) == seq_length:
# # # # #                 input_data = np.expand_dims(np.array(sequence), axis=0)
# # # # #                 y_pred = model.predict(input_data).squeeze()
# # # # #                 i_pred = int(np.argmax(y_pred))
# # # # #                 conf = y_pred[i_pred]

# # # # #                 if conf > 0.9:  # 신뢰도가 90% 이상일 때만 인식
# # # # #                     action = actions[i_pred]
# # # # #                     action_seq.append(action)

# # # # #                     if len(action_seq) > 3:
# # # # #                         action_seq = action_seq[-3:]

# # # # #                     if action_seq.count(action) > 1:
# # # # #                         this_action = action
# # # # #                     else:
# # # # #                         this_action = ' '

# # # # #                     cv2.putText(img, f'{this_action}', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

# # # # #         # 양손의 랜드마크 그리기
# # # # #         for hand_landmarks in result.multi_hand_landmarks:
# # # # #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # # # #     cv2.imshow('Hand Gesture Recognition', img)

# # # # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # #         break

# # # # # cap.release()
# # # # # cv2.destroyAllWindows()


# # # # # 양 손 분리 학습
# # # # # import cv2
# # # # # import mediapipe as mp
# # # # # import numpy as np
# # # # # import tensorflow as tf
# # # # # from PIL import ImageFont, ImageDraw, Image

# # # # # # 학습된 모델 로드
# # # # # left_hand_model = tf.keras.models.load_model('left_hand_model.keras')
# # # # # right_hand_model = tf.keras.models.load_model('right_hand_model.keras')

# # # # # # 인식할 손동작 리스트
# # # # # actions = ['만나다', '좋다', '안녕', '당신', '이름', '무엇']
# # # # # seq_length = 30

# # # # # # MediaPipe Hands 초기화
# # # # # mp_hands = mp.solutions.hands
# # # # # mp_drawing = mp.solutions.drawing_utils
# # # # # hands = mp_hands.Hands(
# # # # #     max_num_hands=2,  # 최대 2개의 손을 인식
# # # # #     min_detection_confidence=0.5,
# # # # #     min_tracking_confidence=0.5
# # # # # )

# # # # # # 웹캠 열기
# # # # # cap = cv2.VideoCapture(1)

# # # # # # 웹캠이 제대로 열리지 않는 경우 다른 인덱스를 시도
# # # # # if not cap.isOpened():
# # # # #     cap = cv2.VideoCapture(0)
# # # # # if not cap.isOpened():
# # # # #     print("웹캠을 열 수 없습니다.")
# # # # #     exit()

# # # # # left_sequence = []
# # # # # right_sequence = []
# # # # # action_seq = []

# # # # # # 한글 폰트 설정 - 절대 경로 사용
# # # # # fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
# # # # # font = ImageFont.truetype(fontpath, 32)

# # # # # while cap.isOpened():
# # # # #     ret, img = cap.read()
# # # # #     if not ret:
# # # # #         break

# # # # #     img = cv2.flip(img, 1)
# # # # #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # # # #     result = hands.process(img_rgb)

# # # # #     if result.multi_hand_landmarks is not None:
# # # # #         for hand_landmarks in result.multi_hand_landmarks:
# # # # #             joint = np.zeros((21, 4))
# # # # #             for j, lm in enumerate(hand_landmarks.landmark):
# # # # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # # # #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # # # #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # # # #             v = v2 - v1
# # # # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # # # #             # 내적 값 클리핑
# # # # #             dot_product = np.einsum('nt,nt->n', v, v)
# # # # #             dot_product = np.clip(dot_product, -1.0, 1.0)

# # # # #             angle = np.arccos(dot_product)
# # # # #             angle = np.degrees(angle)

# # # # #             d = np.concatenate([joint.flatten(), angle])

# # # # #             # 왼손 또는 오른손 시퀀스에 추가
# # # # #             if hand_landmarks.landmark[0].x < 0.5:  # 왼손 (화면 기준)
# # # # #                 left_sequence.append(d)
# # # # #                 if len(left_sequence) > seq_length:
# # # # #                     left_sequence.pop(0)
# # # # #             else:  # 오른손 (화면 기준)
# # # # #                 right_sequence.append(d)
# # # # #                 if len(right_sequence) > seq_length:
# # # # #                     right_sequence.pop(0)

# # # # #             if len(left_sequence) == seq_length and len(right_sequence) == seq_length:
# # # # #                 left_input_data = np.expand_dims(np.array(left_sequence), axis=0)
# # # # #                 right_input_data = np.expand_dims(np.array(right_sequence), axis=0)

# # # # #                 left_y_pred = left_hand_model.predict(left_input_data).squeeze()
# # # # #                 right_y_pred = right_hand_model.predict(right_input_data).squeeze()

# # # # #                 left_i_pred = int(np.argmax(left_y_pred))
# # # # #                 right_i_pred = int(np.argmax(right_y_pred))

# # # # #                 left_conf = left_y_pred[left_i_pred]
# # # # #                 right_conf = right_y_pred[right_i_pred]

# # # # #                 if left_conf > 0.9 and right_conf > 0.9 and left_i_pred == right_i_pred:  # 양 손 모두 같은 동작을 인식
# # # # #                     action = actions[left_i_pred]
# # # # #                     action_seq.append(action)

# # # # #                     if len(action_seq) > 3:
# # # # #                         action_seq = action_seq[-3:]

# # # # #                     if action_seq.count(action) > 1:
# # # # #                         this_action = action
# # # # #                     else:
# # # # #                         this_action = ' '

# # # # #                     # 이미지에 한글 텍스트 추가
# # # # #                     img_pil = Image.fromarray(img)
# # # # #                     draw = ImageDraw.Draw(img_pil)
# # # # #                     draw.text((10, 30), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # # # #                     img = np.array(img_pil)

# # # # #         # 양손의 랜드마크 그리기
# # # # #         for hand_landmarks in result.multi_hand_landmarks:
# # # # #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # # # #     cv2.imshow('Hand Gesture Recognition', img)

# # # # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # #         break

# # # # # cap.release()
# # # # # cv2.destroyAllWindows()


# # # # # # 한글 자막출력 버전 
# # # # # import cv2
# # # # # import mediapipe as mp
# # # # # import numpy as np
# # # # # import tensorflow as tf
# # # # # from PIL import ImageFont, ImageDraw, Image

# # # # # # 학습된 모델 로드
# # # # # model = tf.keras.models.load_model('models/hand_gesture_model.keras')

# # # # # # 인식할 손동작 리스트
# # # # # actions = ['만나다', '좋다', '안녕' '당신', '이름', '무엇']
# # # # # seq_length = 30

# # # # # # MediaPipe Hands 초기화
# # # # # mp_hands = mp.solutions.hands
# # # # # mp_drawing = mp.solutions.drawing_utils
# # # # # hands = mp_hands.Hands(
# # # # #     max_num_hands=2,  # 최대 2개의 손을 인식
# # # # #     min_detection_confidence=0.5,
# # # # #     min_tracking_confidence=0.5
# # # # # )

# # # # # # 웹캠 열기
# # # # # cap = cv2.VideoCapture(1)

# # # # # # 웹캠이 제대로 열리지 않는 경우 다른 인덱스를 시도
# # # # # if not cap.isOpened():
# # # # #     cap = cv2.VideoCapture(0)
# # # # # if not cap.isOpened():
# # # # #     print("웹캠을 열 수 없습니다.")
# # # # #     exit()

# # # # # sequence = []
# # # # # action_seq = []

# # # # # # 한글 폰트 설정 - 절대 경로 사용
# # # # # fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
# # # # # font = ImageFont.truetype(fontpath, 32)

# # # # # while cap.isOpened():
# # # # #     ret, img = cap.read()
# # # # #     if not ret:
# # # # #         break

# # # # #     img = cv2.flip(img, 1)
# # # # #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # # # #     result = hands.process(img_rgb)

# # # # #     if result.multi_hand_landmarks is not None:
# # # # #         for hand_landmarks in result.multi_hand_landmarks:
# # # # #             joint = np.zeros((21, 4))
# # # # #             for j, lm in enumerate(hand_landmarks.landmark):
# # # # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # # # #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # # # #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # # # #             v = v2 - v1
# # # # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # # # #             # 내적 값 클리핑
# # # # #             dot_product = np.einsum('nt,nt->n', v, v)
# # # # #             dot_product = np.clip(dot_product, -1.0, 1.0)

# # # # #             angle = np.arccos(dot_product)
# # # # #             angle = np.degrees(angle)

# # # # #             d = np.concatenate([joint.flatten(), angle])

# # # # #             sequence.append(d)
# # # # #             if len(sequence) > seq_length:
# # # # #                 sequence.pop(0)

# # # # #             if len(sequence) == seq_length:
# # # # #                 input_data = np.expand_dims(np.array(sequence), axis=0)
# # # # #                 y_pred = model.predict(input_data).squeeze()
# # # # #                 i_pred = int(np.argmax(y_pred))
# # # # #                 conf = y_pred[i_pred]

# # # # #                 if conf > 0.9:  # 신뢰도가 90% 이상일 때만 인식
# # # # #                     action = actions[i_pred]
# # # # #                     action_seq.append(action)

# # # # #                     if len(action_seq) > 3:
# # # # #                         action_seq = action_seq[-3:]

# # # # #                     if action_seq.count(action) > 1:
# # # # #                         this_action = action
# # # # #                     else:
# # # # #                         this_action = ' '

# # # # #                     # 이미지에 한글 텍스트 추가
# # # # #                     img_pil = Image.fromarray(img)
# # # # #                     draw = ImageDraw.Draw(img_pil)
# # # # #                     draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # # # #                     img = np.array(img_pil)

# # # # #         # 양손의 랜드마크 그리기
# # # # #         for hand_landmarks in result.multi_hand_landmarks:
# # # # #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # # # #     cv2.imshow('Hand Gesture Recognition', img)

# # # # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # # # #         break

# # # # # cap.release()
# # # # # cv2.destroyAllWindows()


# # # # #양손 사용, 한손 사용 분리
# # # # import cv2
# # # # import mediapipe as mp
# # # # import numpy as np
# # # # import tensorflow as tf
# # # # from PIL import ImageFont, ImageDraw, Image

# # # # # 학습된 모델 로드
# # # # model = tf.keras.models.load_model('models/1_hand_gesture_model.keras')

# # # # # 인식할 손동작 리스트
# # # # actions_two = ['만나다', '좋다', '안녕']
# # # # actions_one = ['당신', '이름', '무엇']
# # # # seq_length = 30

# # # # # MediaPipe Hands 초기화
# # # # mp_hands = mp.solutions.hands
# # # # mp_drawing = mp.solutions.drawing_utils
# # # # hands = mp_hands.Hands(
# # # #     max_num_hands=2,  # 최대 2개의 손을 인식
# # # #     min_detection_confidence=0.5,
# # # #     min_tracking_confidence=0.5
# # # # )

# # # # # 웹캠 열기
# # # # cap = cv2.VideoCapture(1)

# # # # # 웹캠이 제대로 열리지 않는 경우 다른 인덱스를 시도
# # # # if not cap.isOpened():
# # # #     cap = cv2.VideoCapture(0)
# # # # if not cap.isOpened():
# # # #     print("웹캠을 열 수 없습니다.")
# # # #     exit()

# # # # sequence = {'left': [], 'right': []}
# # # # action_seq = []

# # # # # 한글 폰트 설정 - 절대 경로 사용
# # # # fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
# # # # font = ImageFont.truetype(fontpath, 32)

# # # # while cap.isOpened():
# # # #     ret, img = cap.read()
# # # #     if not ret:
# # # #         break

# # # #     img = cv2.flip(img, 1)
# # # #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # # #     result = hands.process(img_rgb)

# # # #     if result.multi_hand_landmarks is not None:
# # # #         found_right_hand = False
# # # #         found_left_hand = False

# # # #         for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# # # #             joint = np.zeros((21, 4))
# # # #             for j, lm in enumerate(hand_landmarks.landmark):
# # # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # # #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # # #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # # #             v = v2 - v1
# # # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # # #             dot_product = np.einsum('nt,nt->n', v, v)
# # # #             dot_product = np.clip(dot_product, -1.0, 1.0)

# # # #             angle = np.arccos(dot_product)
# # # #             angle = np.degrees(angle)

# # # #             d = np.concatenate([joint.flatten(), angle])

# # # #             hand_label = hand_info.classification[0].label
# # # #             if hand_label == 'Right':
# # # #                 sequence['right'].append(d)
# # # #                 found_right_hand = True
# # # #                 if len(sequence['right']) > seq_length:
# # # #                     sequence['right'].pop(0)
# # # #             else:
# # # #                 sequence['left'].append(d)
# # # #                 found_left_hand = True
# # # #                 if len(sequence['left']) > seq_length:
# # # #                     sequence['left'].pop(0)

# # # #         # 양손을 모두 사용하는 경우
# # # #         if found_right_hand and found_left_hand and len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
# # # #             input_data = np.expand_dims(np.array(sequence['right'] + sequence['left']), axis=0)
# # # #             y_pred = model.predict(input_data)
# # # #             y_pred_action = y_pred[0].squeeze()
# # # #             y_pred_both_hands = y_pred[1].squeeze()

# # # #             i_pred = int(np.argmax(y_pred_action))
# # # #             conf = y_pred_action[i_pred]

# # # #             if conf > 0.9:
# # # #                 action = actions_two[i_pred] if y_pred_both_hands >= 0.5 else actions_one[i_pred]
# # # #                 action_seq.append(action)

# # # #                 if len(action_seq) > 3:
# # # #                     action_seq = action_seq[-3:]

# # # #                 if action_seq.count(action) > 1:
# # # #                     this_action = action
# # # #                 else:
# # # #                     this_action = ' '

# # # #                 img_pil = Image.fromarray(img)
# # # #                 draw = ImageDraw.Draw(img_pil)
# # # #                 draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # # #                 img = np.array(img_pil)

# # # #         # 오른손만 사용하는 경우
# # # #         elif found_right_hand and not found_left_hand and len(sequence['right']) == seq_length:
# # # #             input_data = np.expand_dims(np.array(sequence['right']), axis=0)
# # # #             y_pred = model.predict(input_data)
# # # #             y_pred_action = y_pred[0].squeeze()
# # # #             y_pred_both_hands = y_pred[1].squeeze()

# # # #             i_pred = int(np.argmax(y_pred_action))
# # # #             conf = y_pred_action[i_pred]

# # # #             if conf > 0.9:
# # # #                 action = actions_two[i_pred] if y_pred_both_hands >= 0.5 else actions_one[i_pred]
# # # #                 action_seq.append(action)

# # # #                 if len(action_seq) > 3:
# # # #                     action_seq = action_seq[-3:]

# # # #                 if action_seq.count(action) > 1:
# # # #                     this_action = action
# # # #                 else:
# # # #                     this_action = ' '

# # # #                 img_pil = Image.fromarray(img)
# # # #                 draw = ImageDraw.Draw(img_pil)
# # # #                 draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # # #                 img = np.array(img_pil)

# # # #         # 왼손만 사용하는 경우
# # # #         elif found_left_hand and not found_right_hand and len(sequence['left']) == seq_length:
# # # #             input_data = np.expand_dims(np.array(sequence['left']), axis=0)
# # # #             y_pred = model.predict(input_data)
# # # #             y_pred_action = y_pred[0].squeeze()
# # # #             y_pred_both_hands = y_pred[1].squeeze()

# # # #             i_pred = int(np.argmax(y_pred_action))
# # # #             conf = y_pred_action[i_pred]

# # # #             if conf > 0.9:
# # # #                 action = actions_two[i_pred] if y_pred_both_hands >= 0.5 else actions_one[i_pred]
# # # #                 action_seq.append(action)

# # # #                 if len(action_seq) > 3:
# # # #                     action_seq = action_seq[-3:]

# # # #                 if action_seq.count(action) > 1:
# # # #                     this_action = action
# # # #                 else:
# # # #                     this_action = ' '

# # # #                 img_pil = Image.fromarray(img)
# # # #                 draw = ImageDraw.Draw(img_pil)
# # # #                 draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # # #                 img = np.array(img_pil)

# # # #         for hand_landmarks in result.multi_hand_landmarks:
# # # #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # # #     cv2.imshow('Hand Gesture Recognition', img)

# # # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # # #         break

# # # # cap.release()
# # # # cv2.destroyAllWindows()



# # # found 추가 - 차원 불일치 발생
# # # import cv2
# # # import mediapipe as mp
# # # import numpy as np
# # # import tensorflow as tf
# # # from PIL import ImageFont, ImageDraw, Image

# # # # 학습된 모델 로드
# # # model = tf.keras.models.load_model('models/hand_gesture_model.keras')

# # # # 인식할 손동작 리스트
# # # actions = ['meet', 'nice', 'hello', 'you', 'name', 'what']
# # # seq_length = 30

# # # # MediaPipe Hands 초기화
# # # mp_hands = mp.solutions.hands
# # # mp_drawing = mp.solutions.drawing_utils
# # # hands = mp_hands.Hands(
# # #     max_num_hands=2,  # 최대 2개의 손을 인식
# # #     min_detection_confidence=0.5,
# # #     min_tracking_confidence=0.5
# # # )

# # # # 웹캠 열기
# # # cap = cv2.VideoCapture(1)

# # # # 웹캠이 제대로 열리지 않는 경우 다른 인덱스를 시도
# # # if not cap.isOpened():
# # #     cap = cv2.VideoCapture(0)
# # # if not cap.isOpened():
# # #     print("웹캠을 열 수 없습니다.")
# # #     exit()

# # # sequence = {'left': [], 'right': []}
# # # action_seq = []

# # # # 한글 폰트 설정 - 절대 경로 사용
# # # fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
# # # font = ImageFont.truetype(fontpath, 32)

# # # while cap.isOpened():
# # #     ret, img = cap.read()
# # #     if not ret:
# # #         break

# # #     img = cv2.flip(img, 1)
# # #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #     result = hands.process(img_rgb)

# # #     if result.multi_hand_landmarks is not None:
# # #         found_right_hand = False
# # #         found_left_hand = False

# # #         for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# # #             joint = np.zeros((21, 4))
# # #             for j, lm in enumerate(hand_landmarks.landmark):
# # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # #             v = v2 - v1
# # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # #             dot_product = np.einsum('nt,nt->n', v, v)
# # #             dot_product = np.clip(dot_product, -1.0, 1.0)

# # #             angle = np.arccos(dot_product)
# # #             angle = np.degrees(angle)

# # #             d = np.concatenate([joint.flatten(), angle])

# # #             hand_label = hand_info.classification[0].label
# # #             if hand_label == 'Right':
# # #                 sequence['right'].append(d)
# # #                 found_right_hand = True
# # #                 if len(sequence['right']) > seq_length:
# # #                     sequence['right'].pop(0)
# # #             else:
# # #                 sequence['left'].append(d)
# # #                 found_left_hand = True
# # #                 if len(sequence['left']) > seq_length:
# # #                     sequence['left'].pop(0)

# # #         # 양손을 모두 사용하는 경우
# # #         if found_right_hand and found_left_hand and len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
# # #             combined_input_data = np.concatenate([sequence['right'], sequence['left']], axis=-1)
# # #             combined_input_data = np.expand_dims(combined_input_data, axis=0)
# # #             y_pred = model.predict(combined_input_data).squeeze()
# # #             i_pred = int(np.argmax(y_pred))
# # #             conf = y_pred[i_pred]

# # #             if conf > 0.9:
# # #                 action = actions[i_pred]
# # #                 action_seq.append(action)

# # #                 if len(action_seq) > 3:
# # #                     action_seq = action_seq[-3:]

# # #                 if action_seq.count(action) > 1:
# # #                     this_action = action
# # #                 else:
# # #                     this_action = ' '

# # #                 img_pil = Image.fromarray(img)
# # #                 draw = ImageDraw.Draw(img_pil)
# # #                 draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # #                 img = np.array(img_pil)

# # #         # 오른손만 사용하는 경우
# # #         elif found_right_hand and not found_left_hand and len(sequence['right']) == seq_length:
# # #             input_data = np.expand_dims(np.array(sequence['right']), axis=0)
# # #             y_pred = model.predict(input_data).squeeze()
# # #             i_pred = int(np.argmax(y_pred))
# # #             conf = y_pred[i_pred]

# # #             if conf > 0.9:
# # #                 action = actions[i_pred]
# # #                 action_seq.append(action)

# # #                 if len(action_seq) > 3:
# # #                     action_seq = action_seq[-3:]

# # #                 if action_seq.count(action) > 1:
# # #                     this_action = action
# # #                 else:
# # #                     this_action = ' '

# # #                 img_pil = Image.fromarray(img)
# # #                 draw = ImageDraw.Draw(img_pil)
# # #                 draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # #                 img = np.array(img_pil)

# # #         # 왼손만 사용하는 경우
# # #         elif found_left_hand and not found_right_hand and len(sequence['left']) == seq_length:
# # #             input_data = np.expand_dims(np.array(sequence['left']), axis=0)
# # #             y_pred = model.predict(input_data).squeeze()
# # #             i_pred = int(np.argmax(y_pred))
# # #             conf = y_pred[i_pred]

# # #             if conf > 0.9:
# # #                 action = actions[i_pred]
# # #                 action_seq.append(action)

# # #                 if len(action_seq) > 3:
# # #                     action_seq = action_seq[-3:]

# # #                 if action_seq.count(action) > 1:
# # #                     this_action = action
# # #                 else:
# # #                     this_action = ' '

# # #                 img_pil = Image.fromarray(img)
# # #                 draw = ImageDraw.Draw(img_pil)
# # #                 draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # #                 img = np.array(img_pil)

# # #         for hand_landmarks in result.multi_hand_landmarks:
# # #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # #     cv2.imshow('Hand Gesture Recognition', img)

# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()




# # # 양손 사용, 한손 사용 분리
# # # import cv2
# # # import mediapipe as mp
# # # import numpy as np
# # # import tensorflow as tf
# # # from PIL import ImageFont, ImageDraw, Image

# # # # 학습된 모델 로드
# # # model = tf.keras.models.load_model('models/hand_gesture_model.keras')

# # # # 인식할 손동작 리스트
# # # actions_two = ['만나다', '좋다', '안녕']
# # # actions_one = ['당신', '이름', '무엇']
# # # seq_length = 30

# # # # MediaPipe Hands 초기화
# # # mp_hands = mp.solutions.hands
# # # mp_drawing = mp.solutions.drawing_utils
# # # hands = mp_hands.Hands(
# # #     max_num_hands=2,  # 최대 2개의 손을 인식
# # #     min_detection_confidence=0.5,
# # #     min_tracking_confidence=0.5
# # # )

# # # # 웹캠 열기
# # # cap = cv2.VideoCapture(1)

# # # # 웹캠이 제대로 열리지 않는 경우 다른 인덱스를 시도
# # # if not cap.isOpened():
# # #     cap = cv2.VideoCapture(0)
# # # if not cap.isOpened():
# # #     print("웹캠을 열 수 없습니다.")
# # #     exit()

# # # sequence = {'left': [], 'right': []}
# # # action_seq = []

# # # # 한글 폰트 설정 - 절대 경로 사용
# # # fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
# # # font = ImageFont.truetype(fontpath, 32)

# # # while cap.isOpened():
# # #     ret, img = cap.read()
# # #     if not ret:
# # #         break

# # #     img = cv2.flip(img, 1)
# # #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #     result = hands.process(img_rgb)

# # #     if result.multi_hand_landmarks is not None:
# # #         for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# # #             joint = np.zeros((21, 4))
# # #             for j, lm in enumerate(hand_landmarks.landmark):
# # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # #             v = v2 - v1
# # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # #             dot_product = np.einsum('nt,nt->n', v, v)
# # #             dot_product = np.clip(dot_product, -1.0, 1.0)

# # #             angle = np.arccos(dot_product)
# # #             angle = np.degrees(angle)

# # #             d = np.concatenate([joint.flatten(), angle])

# # #             hand_label = hand_info.classification[0].label
# # #             if hand_label == 'Right':
# # #                 sequence['right'].append(d)
# # #                 if len(sequence['right']) > seq_length:
# # #                     sequence['right'].pop(0)
# # #             else:
# # #                 sequence['left'].append(d)
# # #                 if len(sequence['left']) > seq_length:
# # #                     sequence['left'].pop(0)

# # #         # 양손을 모두 사용하는 경우
# # #         if len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
# # #             input_data = np.expand_dims(np.array(sequence['right'] + sequence['left']), axis=0)
# # #             y_pred = model.predict(input_data).squeeze()
# # #             i_pred = int(np.argmax(y_pred))
# # #             conf = y_pred[i_pred]

# # #             if i_pred < len(actions_two) and conf > 0.9:
# # #                 action = actions_two[i_pred]
# # #                 action_seq.append(action)

# # #                 if len(action_seq) > 3:
# # #                     action_seq = action_seq[-3:]

# # #                 if action_seq.count(action) > 1:
# # #                     this_action = action
# # #                 else:
# # #                     this_action = ' '

# # #                 img_pil = Image.fromarray(img)
# # #                 draw = ImageDraw.Draw(img_pil)
# # #                 draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # #                 img = np.array(img_pil)

# # #         # 오른손만 사용하는 경우
# # #         elif len(sequence['right']) == seq_length and len(sequence['left']) < seq_length:
# # #             input_data = np.expand_dims(np.array(sequence['right']), axis=0)
# # #             y_pred = model.predict(input_data).squeeze()
# # #             i_pred = int(np.argmax(y_pred))
# # #             conf = y_pred[i_pred]

# # #             if i_pred < len(actions_one) and conf > 0.9:
# # #                 action = actions_one[i_pred]
# # #                 action_seq.append(action)

# # #                 if len(action_seq) > 3:
# # #                     action_seq = action_seq[-3:]

# # #                 if action_seq.count(action) > 1:
# # #                     this_action = action
# # #                 else:
# # #                     this_action = ' '

# # #                 img_pil = Image.fromarray(img)
# # #                 draw = ImageDraw.Draw(img_pil)
# # #                 draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # #                 img = np.array(img_pil)

# # #         # 왼손만 사용하는 경우
# # #         elif len(sequence['left']) == seq_length and len(sequence['right']) < seq_length:
# # #             input_data = np.expand_dims(np.array(sequence['left']), axis=0)
# # #             y_pred = model.predict(input_data).squeeze()
# # #             i_pred = int(np.argmax(y_pred))
# # #             conf = y_pred[i_pred]

# # #             if i_pred < len(actions_one) and conf > 0.9:
# # #                 action = actions_one[i_pred]
# # #                 action_seq.append(action)

# # #                 if len(action_seq) > 3:
# # #                     action_seq = action_seq[-3:]

# # #                 if action_seq.count(action) > 1:
# # #                     this_action = action
# # #                 else:
# # #                     this_action = ' '
                    

# # #                 img_pil = Image.fromarray(img)
# # #                 draw = ImageDraw.Draw(img_pil)
# # #                 draw.text((450, 600), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # #                 img = np.array(img_pil)

# # #         for hand_landmarks in result.multi_hand_landmarks:
# # #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # #     cv2.imshow('Hand Gesture Recognition', img)

# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()


# # # import cv2
# # # import mediapipe as mp
# # # import numpy as np
# # # import tensorflow as tf
# # # from PIL import ImageFont, ImageDraw, Image
# # # import time

# # # # 학습된 모델 로드
# # # right_model = tf.keras.models.load_model('best_right_model.keras')
# # # left_model = tf.keras.models.load_model('best_left_model.keras')

# # # # 인식할 손동작 리스트
# # # actions_right = ['you', 'name', 'what']
# # # actions_left = ['you', 'name', 'what']
# # # actions_both = ['meet', 'nice', 'hello']

# # # seq_length = 30

# # # # MediaPipe Hands 초기화
# # # mp_hands = mp.solutions.hands
# # # mp_drawing = mp.solutions.drawing_utils
# # # hands = mp_hands.Hands(
# # #     max_num_hands=2,  # 최대 2개의 손을 인식
# # #     min_detection_confidence=0.5,
# # #     min_tracking_confidence=0.5
# # # )

# # # # 웹캠 열기
# # # cap = cv2.VideoCapture(1)
# # # if not cap.isOpened():
# # #     cap = cv2.VideoCapture(0)
# # # if not cap.isOpened():
# # #     print("웹캠을 열 수 없습니다.")
# # #     exit()

# # # sequence = {'left': [], 'right': []}
# # # action_seq = []

# # # # 한글 폰트 설정 - 절대 경로 사용
# # # fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
# # # font = ImageFont.truetype(fontpath, 32)

# # # # 자막 출력 시간 초기화
# # # last_action_time = 0
# # # this_action = ''

# # # while cap.isOpened():
# # #     ret, img = cap.read()
# # #     if not ret:
# # #         break

# # #     img = cv2.flip(img, 1)
# # #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #     result = hands.process(img_rgb)

# # #     current_time = time.time()
    
# # #     if result.multi_hand_landmarks is not None:
# # #         for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# # #             joint = np.zeros((21, 4))
# # #             for j, lm in enumerate(hand_landmarks.landmark):
# # #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# # #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# # #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# # #             v = v2 - v1
# # #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# # #             dot_product = np.einsum('nt,nt->n', v, v)
# # #             dot_product = np.clip(dot_product, -1.0, 1.0)

# # #             angle = np.arccos(dot_product)
# # #             angle = np.degrees(angle)

# # #             d = np.concatenate([joint.flatten(), angle])

# # #             hand_label = hand_info.classification[0].label
# # #             if hand_label == 'Right':
# # #                 sequence['right'].append(d)
# # #                 if len(sequence['right']) > seq_length:
# # #                     sequence['right'].pop(0)
# # #             else:
# # #                 sequence['left'].append(d)
# # #                 if len(sequence['left']) > seq_length:
# # #                     sequence['left'].pop(0)

# # #         # 양손 모델 예측
# # #         if len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
# # #             input_data_right = np.expand_dims(np.array(sequence['right']), axis=0)
# # #             input_data_left = np.expand_dims(np.array(sequence['left']), axis=0)
            
# # #             y_pred_right = right_model.predict(input_data_right).squeeze()
# # #             y_pred_left = left_model.predict(input_data_left).squeeze()
            
# # #             i_pred_right = int(np.argmax(y_pred_right))
# # #             i_pred_left = int(np.argmax(y_pred_left))
            
# # #             conf_right = y_pred_right[i_pred_right]
# # #             conf_left = y_pred_left[i_pred_left]

# # #             if conf_right > 0.5 and conf_left > 0.5:  # 확신 임계값 낮추기
# # #                 if i_pred_right < len(actions_both) and i_pred_left < len(actions_both):
# # #                     action = actions_both[i_pred_right]
# # #                     action_seq.append(action)

# # #                     if len(action_seq) > 3:
# # #                         action_seq = action_seq[-3:]

# # #                     if action_seq.count(action) > 1:
# # #                         this_action = action
# # #                     else:
# # #                         this_action = ' '

# # #                     last_action_time = current_time

# # #                     # 동작 인식 후 시퀀스 초기화
# # #                     sequence = {'left': [], 'right': []}

# # #         # 오른손 모델 예측
# # #         elif len(sequence['right']) == seq_length:
# # #             input_data = np.expand_dims(np.array(sequence['right']), axis=0)
# # #             y_pred = right_model.predict(input_data).squeeze()
# # #             i_pred = int(np.argmax(y_pred))
# # #             conf = y_pred[i_pred]

# # #             if conf > 0.5 and i_pred < len(actions_right):
# # #                 action = actions_right[i_pred]
# # #                 action_seq.append(action)

# # #                 if len(action_seq) > 3:
# # #                     action_seq = action_seq[-3:]

# # #                 if action_seq.count(action) > 1:
# # #                     this_action = action
# # #                 else:
# # #                     this_action = ' '

# # #                 last_action_time = current_time

# # #                 # 동작 인식 후 시퀀스 초기화
# # #                 sequence = {'left': [], 'right': []}

# # #         # 왼손 모델 예측
# # #         elif len(sequence['left']) == seq_length:
# # #             input_data = np.expand_dims(np.array(sequence['left']), axis=0)
# # #             y_pred = left_model.predict(input_data).squeeze()
# # #             i_pred = int(np.argmax(y_pred))
# # #             conf = y_pred[i_pred]

# # #             if conf > 0.5 and i_pred < len(actions_left):
# # #                 action = actions_left[i_pred]
# # #                 action_seq.append(action)

# # #                 if len(action_seq) > 3:
# # #                     action_seq = action_seq[-3:]

# # #                 if action_seq.count(action) > 1:
# # #                     this_action = action
# # #                 else:
# # #                     this_action = ' '

# # #                 last_action_time = current_time

# # #                 # 동작 인식 후 시퀀스 초기화
# # #                 sequence = {'left': [], 'right': []}

# # #         for hand_landmarks in result.multi_hand_landmarks:
# # #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# # #     # 자막 출력 (1초 동안 유지)
# # #     if current_time - last_action_time < 1:
# # #         img_pil = Image.fromarray(img)
# # #         draw = ImageDraw.Draw(img_pil)
# # #         draw.text((300, 500), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# # #         img = np.array(img_pil)

# # #     cv2.imshow('Hand Gesture Recognition', img)

# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()

# # import cv2
# # import numpy as np
# # import tensorflow as tf
# # import mediapipe as mp
# # from PIL import ImageFont, ImageDraw, Image
# # import time

# # # MediaPipe Hands 초기화
# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # hands = mp_hands.Hands(
# #     model_complexity=1,
# #     max_num_hands=2,
# #     min_detection_confidence=0.5,
# #     min_tracking_confidence=0.5
# # )

# # # TensorFlow Lite 모델 로드
# # interpreter = tf.lite.Interpreter(model_path='hand_landmarker.task')
# # interpreter.allocate_tensors()

# # # 모델의 입력 및 출력 텐서 정보 얻기
# # input_details = interpreter.get_input_details()
# # output_details = interpreter.get_output_details()

# # # 웹캠 열기
# # cap = cv2.VideoCapture(0)
# # if not cap.isOpened():
# #     print("웹캠을 열 수 없습니다.")
# #     exit()

# # # 한글 폰트 설정
# # fontpath = "/path/to/NotoSansKR-VariableFont_wght.ttf"  # 폰트 경로
# # font = ImageFont.truetype(fontpath, 32)

# # seq_length = 30
# # actions = ['you', 'name', 'what', 'meet', 'nice', 'hello']
# # sequence = []
# # last_action_time = 0
# # this_action = ''

# # while cap.isOpened():
# #     ret, img = cap.read()
# #     if not ret:
# #         break

# #     img = cv2.flip(img, 1)  # 좌우 반전
# #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     result = hands.process(img_rgb)
# #     current_time = time.time()

# #     if result.multi_hand_landmarks:
# #         for hand_landmarks in result.multi_hand_landmarks:
# #             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# #             joint = np.zeros((21, 4))
# #             for j, lm in enumerate(hand_landmarks.landmark):
# #                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# #             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# #             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# #             v = v2 - v1
# #             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# #             angle = np.arccos(np.clip(np.einsum('nt,nt->n', v, v), -1.0, 1.0))
# #             angle = np.degrees(angle)

# #             d = np.concatenate([joint.flatten(), angle])

# #             sequence.append(d)
# #             if len(sequence) > seq_length:
# #                 sequence.pop(0)

# #             if len(sequence) == seq_length:
# #                 input_data = np.expand_dims(np.array(sequence), axis=0).astype(np.float32)
# #                 interpreter.set_tensor(input_details[0]['index'], input_data)
# #                 interpreter.invoke()
# #                 y_pred = interpreter.get_tensor(output_details[0]['index'])
# #                 i_pred = int(np.argmax(y_pred))
# #                 conf = y_pred[0][i_pred]

# #                 if conf > 0.5:
# #                     this_action = actions[i_pred]
# #                     last_action_time = current_time

# #     # 자막 출력 (1초 동안 유지)
# #     if current_time - last_action_time < 1:
# #         img_pil = Image.fromarray(img)
# #         draw = ImageDraw.Draw(img_pil)
# #         draw.text((300, 500), f'{this_action}', font=font, fill=(255, 255, 255, 0))
# #         img = np.array(img_pil)

# #     cv2.imshow('Hand Gesture Recognition', img)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from PIL import ImageFont, ImageDraw, Image
# import time

# # 학습된 모델 로드
# right_model = tf.keras.models.load_model('best_right_model.keras')
# left_model = tf.keras.models.load_model('best_left_model.keras')

# # 인식할 손동작 리스트
# actions_right = ['you', 'name', 'what']
# actions_left = ['you', 'name', 'what']
# actions_both = ['meet', 'nice', 'hello']

# seq_length = 30

# # MediaPipe Hands 초기화
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=2,  # 최대 2개의 손을 인식
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # 웹캠 열기
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("웹캠을 열 수 없습니다.")
#     exit()

# sequence = {'left': [], 'right': []}
# action_seq = []

# # 한글 폰트 설정 - 절대 경로 사용
# fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
# font = ImageFont.truetype(fontpath, 32)

# # 자막 출력 시간 초기화
# last_action_time = 0
# this_action = ''

# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         break

#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = hands.process(img_rgb)

#     current_time = time.time()
    
#     if result.multi_hand_landmarks is not None:
#         for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
#             joint = np.zeros((21, 4))
#             for j, lm in enumerate(hand_landmarks.landmark):
#                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

#             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
#             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
#             v = v2 - v1
#             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

#             dot_product = np.einsum('nt,nt->n', v, v)
#             dot_product = np.clip(dot_product, -1.0, 1.0)

#             angle = np.arccos(dot_product)
#             angle = np.degrees(angle)

#             d = np.concatenate([joint.flatten(), angle])

#             hand_label = hand_info.classification[0].label
#             if hand_label == 'Right':
#                 sequence['right'].append(d)
#                 if len(sequence['right']) > seq_length:
#                     sequence['right'].pop(0)
#             else:
#                 sequence['left'].append(d)
#                 if len(sequence['left']) > seq_length:
#                     sequence['left'].pop(0)

#         # 양손 모델 예측
#         if len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
#             input_data_right = np.expand_dims(np.array(sequence['right']), axis=0)
#             input_data_left = np.expand_dims(np.array(sequence['left']), axis=0)
            
#             y_pred_right = right_model.predict(input_data_right).squeeze()
#             y_pred_left = left_model.predict(input_data_left).squeeze()
            
#             i_pred_right = int(np.argmax(y_pred_right))
#             i_pred_left = int(np.argmax(y_pred_left))
            
#             conf_right = y_pred_right[i_pred_right]
#             conf_left = y_pred_left[i_pred_left]

#             if conf_right > 0.5 and conf_left > 0.5:  # 확신 임계값 낮추기
#                 if i_pred_right < len(actions_both) and i_pred_left < len(actions_both):
#                     action = actions_both[i_pred_right]
#                     action_seq.append(action)

#                     if len(action_seq) > 3:
#                         action_seq = action_seq[-3:]

#                     if action_seq.count(action) > 1:
#                         this_action = action
#                     else:
#                         this_action = ' '

#                     last_action_time = current_time

#                     # 동작 인식 후 시퀀스 초기화
#                     sequence = {'left': [], 'right': []}

#         # 오른손 모델 예측
#         elif len(sequence['right']) == seq_length:
#             input_data = np.expand_dims(np.array(sequence['right']), axis=0)
#             y_pred = right_model.predict(input_data).squeeze()
#             i_pred = int(np.argmax(y_pred))
#             conf = y_pred[i_pred]

#             if conf > 0.5 and i_pred < len(actions_right):
#                 action = actions_right[i_pred]
#                 action_seq.append(action)

#                 if len(action_seq) > 3:
#                     action_seq = action_seq[-3:]

#                 if action_seq.count(action) > 1:
#                     this_action = action
#                 else:
#                     this_action = ' '

#                 last_action_time = current_time

#                 # 동작 인식 후 시퀀스 초기화
#                 sequence = {'left': [], 'right': []}

#         # 왼손 모델 예측
#         elif len(sequence['left']) == seq_length:
#             input_data = np.expand_dims(np.array(sequence['left']), axis=0)
#             y_pred = left_model.predict(input_data).squeeze()
#             i_pred = int(np.argmax(y_pred))
#             conf = y_pred[i_pred]

#             if conf > 0.5 and i_pred < len(actions_left):
#                 action = actions_left[i_pred]
#                 action_seq.append(action)

#                 if len(action_seq) > 3:
#                     action_seq = action_seq[-3:]

#                 if action_seq.count(action) > 1:
#                     this_action = action
#                 else:
#                     this_action = ' '

#                 last_action_time = current_time

#                 # 동작 인식 후 시퀀스 초기화
#                 sequence = {'left': [], 'right': []}

#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # 자막 출력 (1초 동안 유지)
#     if current_time - last_action_time < 1:
#         img_pil = Image.fromarray(img)
#         draw = ImageDraw.Draw(img_pil)
#         draw.text((300, 500), f'{this_action}', font=font, fill=(255, 255, 255, 0))
#         img = np.array(img_pil)

#     cv2.imshow('Hand Gesture Recognition', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from PIL import ImageFont, ImageDraw, Image
# import time

# # 학습된 모델 로드
# right_model = tf.keras.models.load_model('best_right_model.keras')
# left_model = tf.keras.models.load_model('best_left_model.keras')

# # 인식할 손동작 리스트
# actions_right = ['you', 'name', 'what']
# actions_left = ['you', 'name', 'what']
# actions_both = ['meet', 'nice', 'hello']

# seq_length = 30

# # MediaPipe Hands 초기화
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=2,  # 최대 2개의 손을 인식
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # 웹캠 열기
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("웹캠을 열 수 없습니다.")
#     exit()

# sequence = {'left': [], 'right': []}
# action_seq = []

# # 한글 폰트 설정 - 절대 경로 사용
# fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
# font = ImageFont.truetype(fontpath, 32)

# # 자막 출력 시간 초기화
# last_action_time = 0
# this_action = ''

# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         break

#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = hands.process(img_rgb)

#     current_time = time.time()
    
#     if result.multi_hand_landmarks is not None:
#         for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
#             joint = np.zeros((21, 4))
#             for j, lm in enumerate(hand_landmarks.landmark):
#                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

#             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
#             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
#             v = v2 - v1
#             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

#             dot_product = np.einsum('nt,nt->n', v, v)
#             dot_product = np.clip(dot_product, -1.0, 1.0)

#             angle = np.arccos(dot_product)
#             angle = np.degrees(angle)

#             d = np.concatenate([joint.flatten(), angle])

#             hand_label = hand_info.classification[0].label
#             if hand_label == 'Right':
#                 sequence['right'].append(d)
#                 if len(sequence['right']) > seq_length:
#                     sequence['right'].pop(0)
#             else:
#                 sequence['left'].append(d)
#                 if len(sequence['left']) > seq_length:
#                     sequence['left'].pop(0)

            

#         # 양손 모델 예측
#         if len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
#             input_data_right = np.expand_dims(np.array(sequence['right']), axis=0)
#             input_data_left = np.expand_dims(np.array(sequence['left']), axis=0)
            
#             y_pred_right = right_model.predict(input_data_right).squeeze()
#             y_pred_left = left_model.predict(input_data_left).squeeze()
            
#             i_pred_right = int(np.argmax(y_pred_right))
#             i_pred_left = int(np.argmax(y_pred_left))
            
#             conf_right = y_pred_right[i_pred_right]
#             conf_left = y_pred_left[i_pred_left]

#             if conf_right > 0.5 and conf_left > 0.5:  # 확신 임계값 낮추기
#                 if i_pred_right < len(actions_both) and i_pred_left < len(actions_both):
#                     action = actions_both[i_pred_right]
#                     action_seq.append(action)

#                     if len(action_seq) > 3:
#                         action_seq = action_seq[-3:]

#                     if action_seq.count(action) > 1:
#                         this_action = action
#                     else:
#                         this_action = ' '

#                     last_action_time = current_time

#                     # 동작 인식 후 시퀀스 초기화
#                     sequence = {'left': [], 'right': []}

#         # 오른손 모델 예측
#         elif len(sequence['right']) == seq_length:
#             input_data = np.expand_dims(np.array(sequence['right']), axis=0)
#             y_pred = right_model.predict(input_data).squeeze()
#             i_pred = int(np.argmax(y_pred))
#             conf = y_pred[i_pred]

#             if conf > 0.5 and i_pred < len(actions_right):
#                 action = actions_right[i_pred]
#                 action_seq.append(action)

#                 if len(action_seq) > 3:
#                     action_seq = action_seq[-3:]

#                 if action_seq.count(action) > 1:
#                     this_action = action
#                 else:
#                     this_action = ' '

#                 last_action_time = current_time

#                 # 동작 인식 후 시퀀스 초기화
#                 sequence = {'left': [], 'right': []}

#         # 왼손 모델 예측
#         elif len(sequence['left']) == seq_length:
#             input_data = np.expand_dims(np.array(sequence['left']), axis=0)
#             y_pred = left_model.predict(input_data).squeeze()
#             i_pred = int(np.argmax(y_pred))
#             conf = y_pred[i_pred]

#             if conf > 0.5 and i_pred < len(actions_left):
#                 action = actions_left[i_pred]
#                 action_seq.append(action)

#                 if len(action_seq) > 3:
#                     action_seq = action_seq[-3:]

#                 if action_seq.count(action) > 1:
#                     this_action = action
#                 else:
#                     this_action = ' '

#                 last_action_time = current_time

#                 # 동작 인식 후 시퀀스 초기화
#                 sequence = {'left': [], 'right': []}

#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # 자막 출력 (1초 동안 유지)
#     if current_time - last_action_time < 1:
#         img_pil = Image.fromarray(img)
#         draw = ImageDraw.Draw(img_pil)
#         draw.text((300, 500), f'{this_action}', font=font, fill=(255, 255, 255, 0))
#         img = np.array(img_pil)

#     cv2.imshow('Hand Gesture Recognition', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from PIL import ImageFont, ImageDraw, Image
# import time

# # 학습된 모델 로드
# right_model = tf.keras.models.load_model('best_right_model.keras')
# left_model = tf.keras.models.load_model('best_left_model.keras')

# # 인식할 손동작 리스트
# actions_right = ['meet', 'nice', 'hello', 'you', 'name', 'what']
# actions_left = ['meet', 'nice', 'hello', 'you', 'name', 'what']
# actions_both = ['meet', 'nice', 'hello']

# seq_length = 30

# # MediaPipe Hands 초기화
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=2,  # 최대 2개의 손을 인식
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # 웹캠 열기
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("웹캠을 열 수 없습니다.")
#     exit()

# sequence = {'left': [], 'right': []}
# action_seq = []

# # 한글 폰트 설정 - 절대 경로 사용
# fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
# font = ImageFont.truetype(fontpath, 32)

# # 자막 출력 시간 초기화
# last_action_time = 0
# this_action = ''

# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         break

#     img = cv2.flip(img, 1)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = hands.process(img_rgb)

#     current_time = time.time()
    
#     if result.multi_hand_landmarks is not None:
#         for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
#             joint = np.zeros((21, 4))
#             for j, lm in enumerate(hand_landmarks.landmark):
#                 joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

#             v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
#             v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
#             v = v2 - v1
#             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

#             dot_product = np.einsum('nt,nt->n', v, v)
#             dot_product = np.clip(dot_product, -1.0, 1.0)

#             angle = np.arccos(dot_product)
#             angle = np.degrees(angle)

#             d = np.concatenate([joint.flatten(), angle])

#             hand_label = hand_info.classification[0].label
#             if hand_label == 'Right':
#                 sequence['right'].append(d)
#                 if len(sequence['right']) > seq_length:
#                     sequence['right'].pop(0)
#             else:
#                 sequence['left'].append(d)
#                 if len(sequence['left']) > seq_length:
#                     sequence['left'].pop(0)

#         # 양손 모델 예측
#         if len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
#             input_data_right = np.expand_dims(np.array(sequence['right']), axis=0)
#             input_data_left = np.expand_dims(np.array(sequence['left']), axis=0)
            
#             y_pred_right = right_model.predict(input_data_right).squeeze()
#             y_pred_left = left_model.predict(input_data_left).squeeze()
            
#             i_pred_right = int(np.argmax(y_pred_right))
#             i_pred_left = int(np.argmax(y_pred_left))
            
#             conf_right = y_pred_right[i_pred_right]
#             conf_left = y_pred_left[i_pred_left]

#             if i_pred_right < len(actions_right):
#                 print(f"Right hand prediction: {actions_right[i_pred_right]} ({conf_right:.2f})")
#             else:
#                 print(f"Right hand prediction index {i_pred_right} is out of range for actions_right")

#             if i_pred_left < len(actions_left):
#                 print(f"Left hand prediction: {actions_left[i_pred_left]} ({conf_left:.2f})")
#             else:
#                 print(f"Left hand prediction index {i_pred_left} is out of range for actions_left")

#             if conf_right > 0.5 and conf_left > 0.5:
#                 if i_pred_right < len(actions_both) and i_pred_left < len(actions_both):
#                     action = actions_both[i_pred_right]
#                     action_seq.append(action)

#                     if len(action_seq) > 3:
#                         action_seq = action_seq[-3:]

#                     if action_seq.count(action) > 1:
#                         this_action = action
#                     else:
#                         this_action = ' '

#                     last_action_time = current_time

#                     # 동작 인식 후 시퀀스 초기화
#                     sequence = {'left': [], 'right': []}

#         # 오른손 모델 예측
#         elif len(sequence['right']) == seq_length:
#             input_data = np.expand_dims(np.array(sequence['right']), axis=0)
#             y_pred = right_model.predict(input_data).squeeze()
#             i_pred = int(np.argmax(y_pred))
#             conf = y_pred[i_pred]

#             if i_pred < len(actions_right):
#                 print(f"Right hand prediction: {actions_right[i_pred]} ({conf:.2f})")
                
#                 if conf > 0.5:
#                     action = actions_right[i_pred]
#                     action_seq.append(action)

#                     if len(action_seq) > 3:
#                         action_seq = action_seq[-3:]

#                     if action_seq.count(action) > 1:
#                         this_action = action
#                     else:
#                         this_action = ' '

#                     last_action_time = current_time

#                     # 동작 인식 후 시퀀스 초기화
#                     sequence = {'left': [], 'right': []}
#             else:
#                 print(f"Right hand prediction index {i_pred} is out of range for actions_right")

#         # 왼손 모델 예측
#         elif len(sequence['left']) == seq_length:
#             input_data = np.expand_dims(np.array(sequence['left']), axis=0)
#             y_pred = left_model.predict(input_data).squeeze()
#             i_pred = int(np.argmax(y_pred))
#             conf = y_pred[i_pred]

#             if i_pred < len(actions_left):
#                 print(f"Left hand prediction: {actions_left[i_pred]} ({conf:.2f})")

#                 if conf > 0.5:
#                     action = actions_left[i_pred]
#                     action_seq.append(action)

#                     if len(action_seq) > 3:
#                         action_seq = action_seq[-3:]

#                     if action_seq.count(action) > 1:
#                         this_action = action
#                     else:
#                         this_action = ' '

#                     last_action_time = current_time

#                     # 동작 인식 후 시퀀스 초기화
#                     sequence = {'left': [], 'right': []}
#             else:
#                 print(f"Left hand prediction index {i_pred} is out of range for actions_left")

#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # 자막 출력 (1초 동안 유지)
#     if current_time - last_action_time < 1:
#         img_pil = Image.fromarray(img)
#         draw = ImageDraw.Draw(img_pil)
#         draw.text((300, 500), f'{this_action}', font=font, fill=(255, 255, 255, 0))
#         img = np.array(img_pil)

#     cv2.imshow('Hand Gesture Recognition', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import time

# 학습된 모델 로드
right_model = tf.keras.models.load_model('best_right_model.keras')
left_model = tf.keras.models.load_model('best_left_model.keras')

# 인식할 손동작 리스트
actions_right = ['meet', 'nice', 'hello', 'you', 
                 'name', 'what', 'have', 'do not have', 'me']
actions_left = ['meet', 'nice', 'hello', 'you', 
                'name', 'what', 'have', 'do not have', 'me']
actions_both = ['meet', 'nice', 'hello']

seq_length = 30

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
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

sequence = {'left': [], 'right': []}
action_seq = []

# 한글 폰트 설정 - 절대 경로 사용
fontpath = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
font = ImageFont.truetype(fontpath, 32)

# 자막 출력 시간 초기화
last_action_time = 0
this_action = ''
last_saved_action = ''
file_index = 0

# 텍스트 파일 초기화
output_file = open(f'output_{file_index}.txt', 'w')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    current_time = time.time()
    
    if result.multi_hand_landmarks is not None:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            joint = np.zeros((21, 4))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            dot_product = np.einsum('nt,nt->n', v, v)
            dot_product = np.clip(dot_product, -1.0, 1.0)

            angle = np.arccos(dot_product)
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])

            hand_label = hand_info.classification[0].label
            if hand_label == 'Right':
                sequence['right'].append(d)
                if len(sequence['right']) > seq_length:
                    sequence['right'].pop(0)
            else:
                sequence['left'].append(d)
                if len(sequence['left']) > seq_length:
                    sequence['left'].pop(0)

        # 양손 모델 예측
        if len(sequence['right']) == seq_length and len(sequence['left']) == seq_length:
            input_data_right = np.expand_dims(np.array(sequence['right']), axis=0)
            input_data_left = np.expand_dims(np.array(sequence['left']), axis=0)
            
            y_pred_right = right_model.predict(input_data_right).squeeze()
            y_pred_left = left_model.predict(input_data_left).squeeze()
            
            i_pred_right = int(np.argmax(y_pred_right))
            i_pred_left = int(np.argmax(y_pred_left))
            
            conf_right = y_pred_right[i_pred_right]
            conf_left = y_pred_left[i_pred_left]

            if i_pred_right < len(actions_right):
                print(f"Right hand prediction: {actions_right[i_pred_right]} ({conf_right:.2f})")
            else:
                print(f"Right hand prediction index {i_pred_right} is out of range for actions_right")

            if i_pred_left < len(actions_left):
                print(f"Left hand prediction: {actions_left[i_pred_left]} ({conf_left:.2f})")
            else:
                print(f"Left hand prediction index {i_pred_left} is out of range for actions_left")

            if conf_right > 0.5 and conf_left > 0.5:
                if i_pred_right < len(actions_both) and i_pred_left < len(actions_both):
                    action = actions_both[i_pred_right]
                    action_seq.append(action)

                    if len(action_seq) > 3:
                        action_seq = action_seq[-3:]

                    if action_seq.count(action) > 1:
                        this_action = action
                    else:
                        this_action = ' '

                    last_action_time = current_time

                    # 동작 인식 후 시퀀스 초기화
                    sequence = {'left': [], 'right': []}

        # 오른손 모델 예측
        elif len(sequence['right']) == seq_length:
            input_data = np.expand_dims(np.array(sequence['right']), axis=0)
            y_pred = right_model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if i_pred < len(actions_right):
                print(f"Right hand prediction: {actions_right[i_pred]} ({conf:.2f})")
                
                if conf > 0.5:
                    action = actions_right[i_pred]
                    action_seq.append(action)

                    if len(action_seq) > 3:
                        action_seq = action_seq[-3:]

                    if action_seq.count(action) > 1:
                        this_action = action
                    else:
                        this_action = ' '

                    last_action_time = current_time

                    # 동작 인식 후 시퀀스 초기화
                    sequence = {'left': [], 'right': []}
            else:
                print(f"Right hand prediction index {i_pred} is out of range for actions_right")

        # 왼손 모델 예측
        elif len(sequence['left']) == seq_length:
            input_data = np.expand_dims(np.array(sequence['left']), axis=0)
            y_pred = left_model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if i_pred < len(actions_left):
                print(f"Left hand prediction: {actions_left[i_pred]} ({conf:.2f})")

                if conf > 0.5:
                    action = actions_left[i_pred]
                    action_seq.append(action)

                    if len(action_seq) > 3:
                        action_seq = action_seq[-3:]

                    if action_seq.count(action) > 1:
                        this_action = action
                    else:
                        this_action = ' '

                    last_action_time = current_time

                    # 동작 인식 후 시퀀스 초기화
                    sequence = {'left': [], 'right': []}
            else:
                print(f"Left hand prediction index {i_pred} is out of range for actions_left")

        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 자막 출력 (1초 동안 유지)
    if current_time - last_action_time < 1:
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((300, 500), f'{this_action}', font=font, fill=(255, 255, 255, 0))
        img = np.array(img_pil)

        # 자막을 텍스트 파일로 저장
        if this_action != last_saved_action:
            output_file.write(f'{this_action}\n')
            last_saved_action = this_action

    cv2.imshow('Hand Gesture Recognition', img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('n'):
        file_index += 1
        output_file.close()
        output_file = open(f'output_{file_index}.txt', 'w')
        last_saved_action = ''

cap.release()
cv2.destroyAllWindows()
output_file.close()