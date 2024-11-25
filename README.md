## hand_gesture_direct
웹캡을 실행시켜 화면에 인식되는 각 손에 좌표를 적용하고 손동작 좌표의 시계열 데이터 직접 추출
OpenCV/MediaPipe를 사용하여 손 좌표 데이터를 추출
Python과 TensorFlow/Keras를 활용하여 수어 데이터를 학습시키고 테스트하는 시스템

## 기능
- 데이터 추출:
  - MediaPipe를 활용하여 영상에서 양손의 좌표 데이터를 추출
  - 손목, 손가락 끝 등 21개의 랜드마크 좌표를 활용
  - 동작의 변화 .npy 형식으로 좌표 저장
- 모델 학습:
  - 좌표 데이터를 입력으로 받아 수어를 인식하는 모델 학습
  - TensorFlow를 사용하여 학습 및 검증
- 실시간 테스트:
  - OpenCV를 통해 실시간으로 손 동작을 인식하고 결과를 출력
  - 학습된 모델을 활용한 예측 결과를 시각화

## 각 파일 설명
- 양손train.ipynb:
  - 모델 학습 및 검증을 수행하는 Jupyter Notebook 파일
  - 데이터 전처리, 모델 설계, 학습, 결과 시각화 포함
- 직접추출.py:
  - MediaPipe를 활용해 손 좌표 데이터를 추출하는 스크립트
  - 비디오 또는 실시간 웹캠 영상을 처리하여 좌표 데이터 저장
- 직접테스트.py:
  - 학습된 모델을 사용해 실시간으로 수어를 인식하고 테스트하는 스크립트
  - OpenCV를 활용한 영상 입력 및 예측 결과 시각화
 
## 기술
	•	Python: 데이터 처리 및 모델 구현
	•	MediaPipe: 손 좌표 추출
	•	TensorFlow/Keras: 딥러닝 모델 설계 및 학습
	•	OpenCV: 실시간 영상 처리
## 결과 테스트
![hello](https://github.com/user-attachments/assets/093a1516-bbf7-402a-8b7f-d48a38c94455)
![meet](https://github.com/user-attachments/assets/b413523b-64dd-4872-839e-32b1adf1a093)
![nice](https://github.com/user-attachments/assets/3f9bbfb9-e8e7-4c31-8f84-44da0c188910)

