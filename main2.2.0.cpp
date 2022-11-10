//
// Created by ljx on 2022/11/10.
//
#include <iostream>
#include "utils.h"
#include "string.h"
#include "Interpreter.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Expr.hpp>

int main(int argc, char **argv) {
//    std::cout << "Hello, World!" << std::endl;
    if (argc < 3) {
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << "output_path:: output_dir" << std::endl;
        return -1;
    }

    const std::string mnn_path = argv[1];

//    std::shared_ptr<MNN::Interpreter> my_interpreter = std::shared_ptr<MNN::Interpreter>(
//            MNN::Interpreter::createFromFile(mnn_path.c_str()));
//
//    // config
//    MNN::ScheduleConfig config;
//    int num_thread = 4;
//    config.numThread = num_thread;
//    MNN::BackendConfig backendConfig;
//    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
//    config.backendConfig = &backendConfig;
//    int forward = MNN_FORWARD_CPU;
//    config.type = static_cast<MNNForwardType>(forward);

// 从模型文件加载并创建新Module
//    const std::string model_file = "/tmp/mymodule.mnn"; // model file with path
    const std::vector<std::string> input_names{"L", "R"};
    const std::vector<std::string> output_names{"output"};
    MNN::Express::Module::Config mdconfig; // default module config
    std::unique_ptr<MNN::Express::Module> module; // module
    module.reset(MNN::Express::Module::load(input_names, output_names, mnn_path.c_str(), &mdconfig));

//    MNN::BackendConfig config;
//    std::shared_ptr<MNN::Express::Executor> executor(Executor::newExecutor(MNN_FORWARD_CPU, config, nthreads));
//    MNN::Express::ExecutorScope scope(executor);
//
//    std::cout << "Dim=" << dim << std::endl;
//    std::cout << "Begin" << std::endl;
//
//    std::unique_ptr<Module> module;
//    Module::Config mdconfig;
//    mdconfig.rearrange = true; // Reduce net buffer memory
//    {
//        AUTOTIME;
//        module.reset(Module::load(input_names, output_names, model_filename.c_str(), &mdconfig));
//    }


// 从现有Module创建新Module，可用于多进程并发
//    std::unique_ptr<MNN::Express::Module> module_shallow_copy;
//    module_shallow_copy.reset(MNN::Express::Module::clone(module.get()));

    // create session
//    MNN::Session *my_session = my_interpreter->createSession(config);
//
//
//    MNN::Tensor *input_tensorL = my_interpreter->getSessionInput(my_session, "L");
//    my_interpreter->resizeTensor(input_tensorL, {1, 3, 400, 640});
//    MNN::Tensor *input_tensorR = my_interpreter->getSessionInput(my_session, "R");
//    my_interpreter->resizeTensor(input_tensorR, {1, 3, 400, 640});
////    my_interpreter->resizeTensor(input_tensor, {1, 3, 416, 416});

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

        std::vector<MNN::Express::VARP> inputs(2);
        inputs[0] = _Input({1, 640, 400, 3}, MNN::Express::NHWC, halide_type_of<float>());
        inputs[1] = _Input({1, 640, 400, 3}, MNN::Express::NHWC, halide_type_of<float>());
        std::vector<float> input_pointer = {inputs[0]->writeMap<float>(),
                                           inputs[1]->writeMap<float>(),
        };

//        for (int i = 0; i < inputs[0]->getInfo()->size; ++i)
//        {
//            input_pointer[0] = imageL.data;
//            input_pointer[1] = imageR.data;
//        }

        std::vector<MNN::Express::VARP> outputs  = module->onForward(inputs);
        auto output_ptr = outputs[0]->readMap<float>();
        std::cout<<"output:"<<output_ptr<<std::endl;
//        std::shared_ptr<MNN::CV::ImageProcess> pretreat(
//                MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::RGB, mean_vals, 3, norm_vals, 3));
//        pretreat->convert(imageL.data, 640, 400, imageL.step[0], input_tensorL);
//        pretreat->convert(imageR.data, 640, 400, imageR.step[0], input_tensorR);
//
//        my_interpreter->runSession(my_session);
//
//        auto output = my_interpreter->getSessionOutput(my_session, "output");
//        auto t_host = new MNN::Tensor(output, MNN::Tensor::CAFFE);
//        output->copyToHostTensor(t_host);
//
//        float *outputdims = t_host->host<float>();
//        std::cout<<"output:"<<output<<std::endl;
    }




    return 0;
}