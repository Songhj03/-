# app.py (Python 코드)
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # 이미지를 OpenCV로 열고 YOLOv4로 객체 감지
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # YOLOv4 모델과 가중치 파일 경로
        config_path = 'yolov4.cfg'
        weights_path = 'yolov4.weights'

        # YOLOv4 모델 로드
        net = cv2.dnn.readNet(weights_path, config_path)

        # 이미지를 객체 감지용으로 변환
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # 객체 감지 수행
        layer_names = net.getUnconnectedOutLayersNames()
        detections = net.forward(layer_names)

        # 결과 이미지에 객체 감지 정보 그리기
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(obj[0] * image.shape[1])
                    center_y = int(obj[1] * image.shape[0])
                    width = int(obj[2] * image.shape[1])
                    height = int(obj[3] * image.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # 객체 박스 그리기
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # 결과 이미지를 저장하거나 웹 페이지에 표시
        result_path = 'result.jpg'
        cv2.imwrite(result_path, image)

        return jsonify({'result_img': result_path})

if __name__ == '__main__':
    app.run(debug=True)