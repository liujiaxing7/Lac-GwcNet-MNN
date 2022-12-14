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


#define STB_IMAGE_IMPLEMENTATION
//#define STBI_NO_STDIO
#include "stb_image.h"

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << "output_path:: output_dir" << std::endl;
        return -1;
    }

    const std::string mnn_path = argv[1];

    const std::vector<std::string> input_names{"L", "R"};
    const std::vector<std::string> output_names{"output"};

//    auto type = MNN_FORWARD_OPENCL;
//    MNN::ScheduleConfig Sconfig;
//    Sconfig.type      = type;
//    Sconfig.numThread = 4;
//    Sconfig.backupType = type;
//    MNN::BackendConfig backendConfig;
//    int precision = MNN::BackendConfig::Precision_Normal;
//    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
//    Sconfig.backendConfig     = &backendConfig;

    MNN::Express::Module::Config mConfig;
//    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(Sconfig));
//    std::shared_ptr<MNN::Express::Module> net(
//            MNN::Express::Module::load(input_names, output_names, mnn_path.c_str(),rtmgr,  &mConfig));
    std::unique_ptr<MNN::Express::Module> module;
    module.reset(MNN::Express::Module::load(input_names, output_names, mnn_path.c_str(), &mConfig));

    auto info = module->getInfo();


    std::string imagespath = argv[2];

    std::vector<std::string> limg;
    std::vector<std::string> rimg;


    ReadImages(imagespath, limg, rimg);


    for (size_t i = 0; i < limg.size(); ++i) {

        int w = 320;
        int h = 320;
        int c = 3;
//        std::vector<MNN::Express::VARP> inputs(2);
//        inputs[0] =  MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NC4HW4);
//        inputs[1] =  MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NC4HW4);
        auto inputLeft = MNN::Express::_Input({1, 3, h, w}, MNN::Express::NC4HW4, halide_type_of<float>());
//        auto inputRight = MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NC4HW4, halide_type_of<float>());
        //auto other = MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NC4HW4);

        auto imgL = limg.at(i);
        auto imgR = rimg.at(i);


        int width, height, channel;
        auto imageL = stbi_load(imgL.c_str(), &width, &height, &channel, 4);
        auto imageR = stbi_load(imgR.c_str(), &width, &height, &channel, 4);

        //std::cout<<imageR<<std::endl;
//        cv::Mat gray1_mat(400, 640, CV_8UC3, imageR);
//        imshow("??????????????????", gray1_mat);
//        cv::waitKey();

        std::cout << "read images" << std::endl;

//

        MNN::CV::Matrix trans;
        trans.setScale((float)(width-1) / (w-1), (float)(height-1) / (h-1));

        MNN::CV::ImageProcess::Config config;
        config.filterType = MNN::CV::BILINEAR;

        float mean[3] = {103.94f, 116.78f, 123.68f};
        float normals[3] = {0.017f, 0.017f, 0.017f};

        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(mean));

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config));
        pretreat->setMatrix(trans);

        pretreat->convert((uint8_t *) imageL, width, height, 0, inputLeft->writeMap<float>() + 0 * 4 * w * h , w, h,
                          4, 0, halide_type_of<float>());

//        pretreat->convert((uint8_t *) imageR, 640, 400, 0, inputRight->writeMap<float>() + 10 * 4 * 400 * 640 , 640, 400,
//                         4, 0, halide_type_of<float>());

        std::cout << "forward" << std::endl;
//
//        MNN::Express::Executor::getGlobalExecutor()->resetProfile();

        auto outputs = module->onForward({inputLeft, inputLeft});
//        MNN::Express::Executor::getGlobalExecutor()->dumpProfile();
//        std::cout << "success" << std::endl;

    }
    return 0;
}