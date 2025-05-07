from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
import cv2
import csv
import torch
import sys
import os
import re
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

from pathlib import Path
MODEL_PATH = Path(__file__).resolve().parent / "pothole.pt"

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

print(f"[DEBUG] 모델 경로: {MODEL_PATH}")
# UI 파일 경로 설정
UI_PATH = os.path.join(os.getcwd(), "res", "mainWindow.ui")
if not os.path.exists(UI_PATH):
    raise FileNotFoundError(f"UI 파일을 찾을 수 없습니다: {UI_PATH}")

Form, Window = uic.loadUiType(UI_PATH)

class MainApp:
    def __init__(self):
        self.app = QApplication([])
        self.window = Window()
        self.form = Form()
        self.form.setupUi(self.window)
        self.data_file_path = os.path.join(os.getcwd(), "data.csv")

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(None, "오류", "웹캠을 열 수 없습니다.")
            sys.exit()

        # YOLOv5 모델 로드
        self.device = select_device("cpu")
        if not MODEL_PATH.exists():
            QMessageBox.critical(None, "오류", f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
            sys.exit()

        self.model = DetectMultiBackend(str(MODEL_PATH), device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.imgsz = (640, 640)

        # 웹캠 프레임 표시 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 버튼 이벤트 연결
        self.form.btnPhoto.clicked.connect(self.capture_photo)
        self.form.btnSave.clicked.connect(self.save_data)
        self.form.btnDel.clicked.connect(self.clear_fields)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            img = cv2.resize(frame, self.imgsz)
            img = img[:, :, ::-1].copy().transpose(2, 0, 1)  # ← copy() 추가
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            if len(img.shape) == 3:
                img = img[None]

            pred = self.model(img)
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in det:
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.form.label_4.setPixmap(QPixmap.fromImage(q_image))

    def capture_photo(self):
        name = self.form.lineEdit_name.text().strip()
        if not name:
            QMessageBox.warning(None, "경고", "이름을 입력하세요!")
            return

        safe_name = re.sub(r'[\\/:*?"<>| ]+', '_', name)
        photos_dir = os.path.join(os.getcwd(), "photos")
        os.makedirs(photos_dir, exist_ok=True)
        filename = os.path.join(photos_dir, f"{safe_name}.png")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            QMessageBox.critical(None, "오류", "웹캠 프레임을 읽을 수 없습니다!")
            return

        # ▶ YOLO 추론도 실행해서 바운딩박스 표시
        img = cv2.resize(frame, self.imgsz)
        img = img[:, :, ::-1].copy().transpose(2, 0, 1)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if len(img.shape) == 3:
            img = img[None]

        pred = self.model(img)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = f"{self.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 저장
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.save(filename)

            self.form.lineEditPhoto.setText(filename)
            QMessageBox.information(None, "저장 완료", f"사진이 저장되었습니다:\n{filename}")
        except Exception as e:
            QMessageBox.critical(None, "오류", f"사진 저장 중 오류가 발생했습니다:\n{e}")

    def save_data(self):
        # ⭐ 저장 시작할 때 포커스를 강제로 빼자
        self.window.focusWidget().clearFocus()

        number = self.form.lineEdit_number.text().strip()
        recommend = self.form.textEdit_recommend.toPlainText().strip()
        name = self.form.lineEdit_name.text().strip()
        photo = self.form.lineEditPhoto.text().strip() if self.form.lineEditPhoto.text().strip() else "N/A"

        if not number or not recommend or not name:
            QMessageBox.warning(None, "입력 오류", "모든 정보를 입력하세요!")
            return

        try:
            with open(self.data_file_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([number, recommend, name, photo])

            QMessageBox.information(None, "저장 완료", f"데이터가 성공적으로 저장되었습니다!\n파일 경로: {self.data_file_path}")
        except Exception as e:
            QMessageBox.critical(None, "저장 실패", f"오류가 발생했습니다:\n{e}")

    def clear_fields(self):
        self.form.lineEdit_name.clear()
        self.form.lineEdit_number.clear()
        self.form.textEdit_recommend.clear()
        self.form.lineEditPhoto.clear()

    def run(self):
        self.window.show()
        self.app.exec()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MainApp()
    app.run()
