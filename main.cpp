#include <iostream>
#include "utils.h"
#include "string.h"
#include "Interpreter.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ImageProcess.hpp>

int main(int argc, char **argv) {
//    std::cout << "Hello, World!" << std::endl;
    if (argc < 3) {
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << "output_path:: output_dir" << std::endl;
        return -1;
    }

    const std::string mnn_path = argv[1];
    std::shared_ptr<MNN::Interpreter> my_interpreter = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromFile(mnn_path.c_str()));

    // config
    MNN::ScheduleConfig config;
    int num_thread = 4;
    config.numThread = num_thread;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;
//    int forward = MNN_FORWARD_CPU;
//    config.type = static_cast<MNNForwardType>(forward);

    // create session
    MNN::Session *my_session = my_interpreter->createSession(config);

    MNN::Tensor *input_tensorL = my_interpreter->getSessionInput(my_session, "L");
    my_interpreter->resizeTensor(input_tensorL, {1, 3, 400, 640});
    MNN::Tensor *input_tensorR = my_interpreter->getSessionInput(my_session, "R");
    my_interpreter->resizeTensor(input_tensorR, {1, 3, 400, 640});
//    my_interpreter->resizeTensor(input_tensor, {1, 3, 416, 416});

    std::string imagespath = argv[2];

    std::vector<std::string> limg;
    std::vector<std::string> rimg;

    ReadImages(imagespath, limg, rimg);

    for (size_t i =0 ; i < limg.size(); ++i)
    {
        auto imgL = limg.at(i);
        auto imgR = rimg.at(i);

        cv::Mat imginL = cv::imread(imgL);
        cv::Mat imginR = cv::imread(imgR);
        cv::Mat frameL = cv::Mat(imginL.rows, imginL.cols, CV_8UC3, imginL.data);
        cv::Mat frameR = cv::Mat(imginR.rows, imginR.cols, CV_8UC3, imginR.data);
        cv::Mat imageL, imageR;
        cv::resize(frameL, imageL, cv::Size(640, 400), cv::INTER_LINEAR);
        cv::resize(frameR, imageR, cv::Size(640, 400), cv::INTER_LINEAR);

        const float mean_vals[3] = {0.485, 0.456, 0.406};
        const float norm_vals[3] = {0.229, 0.224, 0.225};

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::RGB, mean_vals, 3, norm_vals, 3));
        pretreat->convert(imageL.data, 640, 400, imageL.step[0], input_tensorL);
        pretreat->convert(imageR.data, 640, 400, imageR.step[0], input_tensorR);

        my_interpreter->runSession(my_session);

        auto output = my_interpreter->getSessionOutput(my_session, "output");
        auto t_host = new MNN::Tensor(output, MNN::Tensor::CAFFE);
        output->copyToHostTensor(t_host);
        std::cout<<"output:"<<output<<std::endl;
    }




    return 0;
}
