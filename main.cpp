#include <iostream>
#include <string>
#include <random>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

struct CamPic {
    cv::Mat frame;
    double fps;
    CamPic(cv::Mat frame, double fps) : frame(frame), fps(fps) {};
    CamPic clone() {
        return CamPic(frame.clone(), fps);
    }
};

enum Type {
    Paper = 0,
    Rock = 1,
    Scissors = 2
};

const std::string DEVICE = "CPU";
const std::string MODEL_PATH = "/home/ayachi/Documents/schoolTask/rps_game/dataset/best.onnx";
std::mutex mtx;
std::condition_variable cond;
bool ready = false;
bool stop_thread = false;

// onnx model
ov::Core core;
std::shared_ptr<ov::Model> model;
ov::CompiledModel compiledModel;
ov::InferRequest inferRequest;

std::vector<std::string> labels = {"Paper", "Rock", "Scissors"};

CamPic cam_pic(cv::Mat(), 0);

void CaptureCamera() {
    cv::VideoCapture cap(0); // 打开默认摄像头
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return;
    }

    double fps = 0;
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    while (true) {
        cv::Mat temp_frame;
        cap >> temp_frame;
        if (temp_frame.empty()) {
            std::cerr << "ERROR: Could not grab a frame" << std::endl;
            break;
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        ++frameCount;
        if (elapsed.count() >= 1.0) {  // 每秒计算一次FPS
            fps = frameCount / elapsed.count();
            frameCount = 0;
            startTime = std::chrono::high_resolution_clock::now();
        }

        CamPic temp_pic(temp_frame, fps);
        {
            std::lock_guard<std::mutex> lock(mtx);
            cam_pic = temp_pic.clone();
            ready = true;
        }

        cond.notify_one();
        if (stop_thread) {
            break;
        }
    }
    cap.release();
}

CamPic GetFrame() {
    std::unique_lock<std::mutex> lock(mtx);
    cond.wait(lock, []{return ready;});
    ready = false;
    return cam_pic.clone();
}

void InitModel() {
    model = core.read_model(MODEL_PATH);
    compiledModel = core.compile_model(model, DEVICE);
    inferRequest = compiledModel.create_infer_request();
}

int AIPolicy(std::string &filename) {
    // 生成一个随机数，0代表Paper，1代表Rock，2代表Scissors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 2);
    int ai_choice = dis(gen);
    switch(ai_choice) {
        case 0:
            std::cout << "ai chan is: Paper" << std::endl;
            filename = "../asset/pic/paper.jpg";
            break;
        case 1:
            std::cout << "ai chan is: Rock" << std::endl;
            filename = "../asset/pic/stone.jpg";
            break;
        case 2:
            std::cout << "ai chan is: Scissors" << std::endl;
            filename = "../asset/pic/Scissors.jpg";
            break;
    }
    return ai_choice;
}

void Preprocess(cv::Mat &frame, int width=640, int height=640) {
    // 图像预处理，将图像转化为640*640大小
    cv::resize(frame, frame, cv::Size(width, height));
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    frame.convertTo(frame, CV_32F, 1.0 / 255.0); // Normalize to [0, 1]
}

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

std::vector<Detection> Postprocess(const ov::Tensor& output_tensor, float conf_threshold, float iou_threshold, int width, int height) {
    const float* data = output_tensor.data<float>();
    const int dimensions = output_tensor.get_shape()[2];  // YOLOv5 的每个检测框的信息维度
    const int num_detections = output_tensor.get_shape()[1];  // YOLOv5 输出的检测框数量

    std::vector<Detection> detections;

    for (int i = 0; i < num_detections; ++i) {
        const float confidence = data[i * dimensions + 4];  // 获取置信度
        if (confidence >= conf_threshold) {
            const int class_id = static_cast<int>(data[i * dimensions + 5]);  // 获取类别 ID
            const float cx = data[i * dimensions + 0] * width;  // 获取边界框中心 x 坐标
            const float cy = data[i * dimensions + 1] * height;  // 获取边界框中心 y 坐标
            const float w = data[i * dimensions + 2] * width;  // 获取边界框宽度
            const float h = data[i * dimensions + 3] * height;  // 获取边界框高度
            const float xmin = cx - w / 2;
            const float ymin = cy - h / 2;
            detections.push_back({class_id, confidence, cv::Rect(xmin, ymin, w, h)});
        }
    }

    // Apply Non-Maximum Suppression (NMS)
    std::vector<Detection> nms_detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (const auto& detection : detections) {
        boxes.push_back(detection.box);
        scores.push_back(detection.confidence);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);

    for (int idx : indices) {
        nms_detections.push_back(detections[idx]);
    }

    return nms_detections;
}

int GetPlayerCham(cv::Mat frame) {
    std::cout << "classifying......" << std::endl;

    // 对图像进行预处理
    const ov::Shape inputShape = model->input().get_shape();
    const size_t height = inputShape[2];
    const size_t width = inputShape[3];
    Preprocess(frame, static_cast<int>(width), static_cast<int>(height));

    // 确保输入张量的大小正确
    auto input_tensor = inferRequest.get_input_tensor();
    if (input_tensor.get_element_type() != ov::element::f32) {
        std::cerr << "Error: Input tensor type mismatch" << std::endl;
        return -1;
    }
    if (frame.total() * frame.elemSize() != input_tensor.get_byte_size()) {
        std::cerr << "Error: Mismatch in data sizes" << std::endl;
        std::cerr << "Expected size: " << input_tensor.get_byte_size() << ", actual size: " << frame.total() * frame.elemSize() << std::endl;
        return -1;
    }
    std::memcpy(input_tensor.data<float>(), frame.data, input_tensor.get_byte_size());

    // 进行推理
    try {
        inferRequest.infer();
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }

    auto output_tensor = inferRequest.get_output_tensor();
    
    // 打印输出张量信息进行调试
    std::cout << "Output tensor shape: ";
    for (const auto& dim : output_tensor.get_shape()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // 获取所有检测框
    auto detections = Postprocess(output_tensor, 0, 0.001, frame.cols, frame.rows);

    // 如果没有检测到任何框
    if (detections.empty()) {
        return -1;
    }

    // 返回置信度最高的检测框的类别
    auto best_detection = std::max_element(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence < b.confidence;
    });

    std::cout << "Best detection: Class ID = " << best_detection->class_id << ", Confidence = " << best_detection->confidence << std::endl;

    return best_detection->class_id;
}



// 0 is lose; 1 is draw; 2 is win
int GetRes(int player, int ai) {
    switch (player - ai)
    {
    case -1:
    case 2:
        return 2;
    case 0:
        return 1;
    default:
        return 0;
    }
}

void GameStart() {
    // 添加计时器
    auto startTime = std::chrono::high_resolution_clock::now();
    cv::namedWindow("camera", cv::WINDOW_NORMAL);
    cv::namedWindow("ai_cham", cv::WINDOW_NORMAL);
    cv::resizeWindow("camera", 640, 480);
    cv::resizeWindow("ai_cham", 640, 480);
    cv::moveWindow("camera", 0, 0);
    cv::moveWindow("ai_cham", 640, 0);
    // 三秒以后跳出循环
    while (true) {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        if (elapsed.count() >= 3.0) {
            break;
        }
        // 获取摄像头图片
        CamPic camPic = GetFrame();
        cv::Mat frame = camPic.frame;
        cv::imshow("camera", frame);
        // 获取ai_cham图片
        cv::Mat ai_cham = cv::imread("../asset/pic/prepare.jpg");
        if (ai_cham.empty()) {
            std::cerr << "ERROR: Could not load ai_cham image" << std::endl;
            return;
        }
        if (elapsed.count() >= 2.0) {
            cv::putText(ai_cham, "Scissors......", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        } else if (elapsed.count() >= 1.0) {
            cv::putText(ai_cham, "Stone......", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        } else {
            cv::putText(ai_cham, "Paper......", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("ai_cham", ai_cham);
        cv::waitKey(1);
    }

    // 获取摄像头图片
    CamPic camPic = GetFrame();
    cv::Mat frame = camPic.frame;
    // 将摄像头画面投入模型中进行分类
    int playerCham = GetPlayerCham(frame);
    if (playerCham == -1) {
        std::cerr << "Error in classification" << std::endl;
        return;
    }
    std::cout << "player cham is: " << labels[playerCham] << std::endl;
    std::string ai_pic_path;
    int aiCham = AIPolicy(ai_pic_path);
    cv::Mat ai_pic = cv::imread(ai_pic_path);
    if (ai_pic.empty()) {
        std::cerr << "ERROR: Could not load ai_pic image" << std::endl;
        return;
    }
    cv::imshow("camera", frame);
    cv::imshow("ai_cham", ai_pic);
    // 等待3秒
    cv::waitKey(500);
    std::this_thread::sleep_for(std::chrono::seconds(3));
    cv::destroyWindow("camera");
    // 显示结果
    int res = GetRes(playerCham, aiCham);
    if (res == 0) {
        std::cout << "ai chan win" << std::endl;
        cv::Mat win_pic = cv::imread("../asset/pic/win.jpg");
        if (win_pic.empty()) {
            std::cerr << "ERROR: Could not load win_pic image" << std::endl;
            return;
        }
        cv::putText(win_pic, "You Lose", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::imshow("ai_cham", win_pic);
    } else if (res == 1) {
        std::cout << "Draw" << std::endl;
        cv::Mat draw_pic = cv::imread("../asset/pic/prepare.jpg");
        if (draw_pic.empty()) {
            std::cerr << "ERROR: Could not load draw_pic image" << std::endl;
            return;
        }
        cv::putText(draw_pic, "Draw", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::imshow("ai_cham", draw_pic);
    } else {
        std::cout << "You win" << std::endl;
        cv::Mat win_pic = cv::imread("../asset/pic/lose.jpg");
        if (win_pic.empty()) {
            std::cerr << "ERROR: Could not load win_pic image" << std::endl;
            return;
        }
        cv::putText(win_pic, "You Win", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::imshow("ai_cham", win_pic);
    }
    if (cv::waitKey(5000) == 27) {
        return;
    }
}

int main() {
    std::cout << "Hello, World!"<< std::endl;
    std::cout << "loading model: " << MODEL_PATH << std::endl;
    InitModel();
    std::thread cameraThread(CaptureCamera); // 创建一个新的线程来运行displayCamera函数
    // 使用cv创建一个窗口
    cv::namedWindow("ai_cham", cv::WINDOW_NORMAL);
    cv::resizeWindow("ai_cham", 640, 480);
    // 打开图片，将图片展示在ai_cam窗口中
    cv::Mat img = cv::imread("../asset/pic/welcome.jpg");
    if (img.empty()) {
        std::cerr << "ERROR: Could not load welcome image" << std::endl;
        return -1;
    }
    cv::putText(img, "Press any key to play with ai chan!!!!", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    cv::imshow("ai_cham", img);
    while (true) {
        // 捕获用户输入
        char key = cv::waitKey(1);
        if (key == 27) { // ESC
            {
                std::lock_guard<std::mutex> lock(mtx);
                stop_thread = true;
            }
            cond.notify_one();
            break;
        }
        if (key == 32) { // 空格键
            GameStart();
            break;
        }
    }

    std::cout << "game_over" << std::endl;
    // 关闭窗口
    cv::destroyAllWindows();

    return 0;
}
