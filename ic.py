import cv2
from darkflow.net.build import TFNet
import numpy as np

options = {
    "model": "cfg/yolo.cfg",
    "load": "bin/yolo.weights",
    "threshold": 0.2,
    "gpu": 0.0,
}

tfnet = TFNet(options)

colors = [tuple(255 * np.random.rand(3)) for x in range(10)]
capture = cv2.VideoCapture("milan.mp4")

while True:
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result["topleft"]["x"], result["topleft"]["y"])
            br = (result["bottomright"]["x"], result["bottomright"]["y"])
            label = result["label"]
            confidence = result["confidence"]
            text = "{}:{:.0f}%".format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 3)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2
            )
        cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
capture.release()
cv2.destroyAllWindows()
