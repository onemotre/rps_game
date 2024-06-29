import cv2
import torch
import numpy as np
import onnxruntime as ort

# Load ONNX model
onnx_model = "./dataset/best.onnx"
session = ort.InferenceSession(onnx_model)

# Label names
labels = ['Paper', 'Rock', 'Scissors']

# Function to preprocess the image
def preprocess(image):
    img = image.copy()
    img = cv2.resize(img, (640, 640))  # 调整图像大小以适应模型的输入尺寸
    img = img.transpose(2, 0, 1)  # 将数据布局从HWC更改为CHW
    img = np.expand_dims(img, axis=0)  # 增加批次维度
    img = img.astype(np.float32) / 255.0  # 图像归一化
    return img

# Non-Maximum Suppression function
def non_max_suppression(predictions, conf_thres=0.001, iou_thres=0.5):
    boxes = predictions[..., :4]
    scores = predictions[..., 4]
    class_probs = predictions[..., 5:]
    class_ids = np.argmax(class_probs, axis=-1)
    final_scores = scores * class_probs[np.arange(len(class_probs)), class_ids]  # 计算最终置信度

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), final_scores.tolist(), conf_thres, iou_thres)
    if len(indices) > 0:
        indices = indices.flatten()
        return predictions[indices], final_scores[indices]
    return [], []

# Function to perform inference
def infer(image):
    image = preprocess(image)
    inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(None, inputs)[0]

    # 执行NMS
    filtered_predictions, final_scores = non_max_suppression(outputs[0], conf_thres=0.5, iou_thres=0.5)

    if len(filtered_predictions) == 0:
        return None, None, None, None, None, None

    # 找到具有最高置信度的检测框
    max_conf = -np.inf
    max_conf_index = -1
    for i, score in enumerate(final_scores):
        if score > max_conf:
            max_conf = score
            max_conf_index = i

    class_probabilities = filtered_predictions[max_conf_index][5:]
    class_index = np.argmax(class_probabilities)
    predict_label = labels[class_index]
    # 获取最高置信度检测框的x, y, w, h
    xc, yc, w, h = filtered_predictions[max_conf_index][:4]
    return predict_label, xc, yc, w, h, max_conf

# Open camera
cap = cv2.VideoCapture(0)

name = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将这一帧图像保存到指定路径
    name += 1
    cv2.imwrite(f"./test/{name}.jpg", frame)

    # Perform inference
    predict_label, xc, yc, w, h, conf = infer(frame)
    if predict_label is not None:
        cv2.rectangle(frame, (int(xc - w/2), int(yc - h/2)), (int(xc + w/2), int(yc + h/2)), (0, 255, 0), 2)
        # 输出预测结果和置信度
        cv2.putText(frame, f"{predict_label} {conf:.2f}", (int(xc - w/2), int(yc - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv5 Detection', frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
