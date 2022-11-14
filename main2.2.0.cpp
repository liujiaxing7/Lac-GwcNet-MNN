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
//    std::cout << "Hello, World!" << std::endl;
    if (argc < 3) {
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << "output_path:: output_dir" << std::endl;
        return -1;
    }

    const std::string mnn_path = argv[1];


//    MNN::ScheduleConfig sConfig;
//    sConfig.type = MNN_FORWARD_AUTO;
    const std::vector<std::string> input_names{"L", "R"};
    const std::vector<std::string> output_names{"output"};
    MNN::Express::Module::Config mdconfig; // default module config
//    mdconfig.backend = (MNN::Express::Module::BackendInfo *) MNN_FORWARD_OPENCL;


//    std::unique_ptr<MNN::Express::Module> module; // module
//    module.reset(MNN::Express::Module::load(input_names, output_names, mnn_path.c_str(), &mdconfig));

    // Give cache full path which must be Readable and writable

    std::shared_ptr<MNN::Express::Module> net(
            MNN::Express::Module::load(input_names, output_names, mnn_path.c_str(), &mdconfig));

    std::string imagespath = argv[2];

    std::vector<std::string> limg;
    std::vector<std::string> rimg;


    ReadImages(imagespath, limg, rimg);


    for (size_t i = 0; i < limg.size(); ++i) {

        auto inputLeft = MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NCHW);
        auto inputRight = MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NCHW);
//        auto other = MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NCHW);

        auto imgL = limg.at(i);
        auto imgR = rimg.at(i);
        int w = 640;
        int h = 400;
        int c = 3;
        auto imageL = stbi_load(imgL.c_str(), &w, &h, &c, 3);
        auto imageR = stbi_load(imgR.c_str(), &w, &h, &c, 3);

//        cv::Mat gray1_mat(400, 640, CV_8UC3, imageR);
//        imshow("去雾图像显示", gray1_mat);
//        cv::waitKey();

        std::cout << "read images" << std::endl;

//
        MNN::CV::ImageProcess::Config config;
        config.filterType = MNN::CV::BILINEAR;
        MNN::CV::Matrix trans;

        float mean[3] = {103.94f, 116.78f, 123.68f};
        float normals[3] = {0.017f, 0.017f, 0.017f};
//
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(mean));
//
//
        std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config));
        pretreat->setMatrix(trans);

        pretreat->convert((uint8_t *) imageL, 640, 400, 0, inputLeft->writeMap<float>() + 1 * 3 * 400 * 640, 640, 400,
                          3, 0, halide_type_of<float>());
        pretreat->convert((uint8_t *) imageR, 640, 400, 0, inputRight->writeMap<float>() + 1 * 3 * 400 * 640, 640, 400,
                          3, 0, halide_type_of<float>());

        stbi_image_free(imageL);
        stbi_image_free(imageR);
        std::cout << "forward" << std::endl;
        try {
            auto outputs = net->onForward({inputLeft, inputRight});
        }
        catch(char *str){

        }

        std::cout << "success" << std::endl;
//        std::vector<MNN::Express::VARP> outputs  = module->onForward(inputs);
//        auto output_ptr = outputs[0]->readMap<float>();
    }


    return 0;
}