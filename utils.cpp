//
// Created by ljx on 2022/11/10.
//

#include "utils.h"
#include "opencv2/opencv.hpp"
bool isContain(std::string str1, std::string str2)
{
    if (str1.find(str2)!= std::string::npos)
    {
        return true;
    } else
    {
        return false;
    }

}
void ReadImages(const std::string imagesPath, std::vector<std::string> &lImg, std::vector<std::string> &rImg)
{
    std::vector<cv::String> allFiles;
    cv::glob(imagesPath, allFiles, false);

    if (allFiles.size() == 0)
    {
        std::cout<< "there in no file;"<<std::endl;
    }
    for (int i =0 ; i < allFiles.size(); ++i)
    {
        if (isContain(allFiles[i],"L"))
        {
            lImg.push_back(allFiles[i]);
        }
        else if (isContain(allFiles[i], "R"))
        {
            rImg.push_back(allFiles[i]);
        }
    }


}