import cv2
import os
import onnxruntime as ort
import numpy as np

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))  # 调整图像大小以适应模型的输入尺寸
    img = img.transpose(2, 0, 1)  # 将数据布局从HWC更改为CHW
    img = np.expand_dims(img, axis=0)  # 增加批次维度
    img = img.astype(np.float32) / 255.0  # 图像归一化
    return img

def main():
    # 加载ONNX模型
    session = ort.InferenceSession('best.onnx')

    # 加载图像
    image_path = './test'
    image_savePath = "./test_result"
    for img_file in os.listdir(image_path):
        print(f"Processing {img_file}")
        img = load_image(os.path.join(image_path, img_file))

        # 进行推断
        inputs = {session.get_inputs()[0].name: img}
        output = session.run(None, inputs)[0]
        print("Output shape:", output.shape)

        # 找到具有最高置信度的检测框
        max_conf = -np.inf
        max_conf_index = -1
        for i, detection in enumerate(output[0]):
            conf = detection[4]  # 置信度通常存储在第五个元素
            if conf > max_conf:
                max_conf = conf
                max_conf_index = i

        # 获取类别索引，通常类别概率紧跟置信度之后
        class_probabilities = output[0][max_conf_index][5:]  # 取出类别概率
        class_index = np.argmax(class_probabilities)
        print("Highest confidence index:", max_conf_index)
        print("Class probabilities:", class_probabilities)
        print("Class index:", class_index)

        # 获取类别名称
        class_names = ['Paper', 'Rock', 'Scissors']
        predicted_class = class_names[class_index]
        print(f"Predicted class: {predicted_class}")

        # 在图片上绘制检测框
        img = cv2.imread(os.path.join(image_path, img_file))
        xc, yc, w, h = output[0][max_conf_index][:4]
        xc, yc, w, h = int(xc * 640), int(yc * 640), int(w * 640), int(h * 640)
        cv2.rectangle(img, (xc - w // 2, yc - h // 2), (xc + w // 2, yc + h // 2), (0, 255, 0), 2)
        cv2.putText(img, predicted_class, (xc - w // 2, yc - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # 将图片保存在./test_result文件夹
        cv2.imwrite(os.path.join(image_savePath, img_file), img)

if __name__ == "__main__":
    main()
