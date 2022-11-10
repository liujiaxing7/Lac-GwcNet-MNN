//
// Created by ljx on 2022/11/10.
//
#include <iostream>
#include <vector>

#ifndef LAC_GWCNET_UTILS_H
#define LAC_GWCNET_UTILS_H


void ReadImages(const std::string imagesPath, std::vector<std::string> &lImg, std::vector<std::string> &rImg);
bool isContain(std::string str1, std::string str2);

#endif //LAC_GWCNET_UTILS_H
