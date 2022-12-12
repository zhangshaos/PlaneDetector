//
// Created by zxm on 2022/12/9.
//


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <glog/logging.h>
#include "Tools.h"
#include "Detector.h"
#include "cv/LoadNpy.h"


//PASS
void testSegmentByLines() {
  cv::Mat clustersMap(5, 5, CV_32S, -1);
  clustersMap.at<int32_t>(2, 1) = 0;
  clustersMap.at<int32_t>(3, 1) = 0;
  clustersMap.at<int32_t>(4, 0) = 1;
  clustersMap.at<int32_t>(4, 1) = 1;
  clustersMap.at<int32_t>(4, 2) = 1;
  clustersMap.at<int32_t>(4, 3) = 1;
  clustersMap.at<int32_t>(4, 4) = 2;
  clustersMap.at<int32_t>(3, 4) = 2;
  clustersMap.at<int32_t>(2, 4) = 2;
  clustersMap.at<int32_t>(3, 2) = 3;
  clustersMap.at<int32_t>(2, 2) = 3;
  clustersMap.at<int32_t>(1, 2) = 3;
  clustersMap.at<int32_t>(1, 1) = 3;
  clustersMap.at<int32_t>(0, 1) = 3;
  clustersMap.at<int32_t>(0, 2) = 3;
  LOG(INFO) << '\n' << clustersMap << '\n';
  zxm::tool::DrawClusters("../dbg/testSegmentByLines0.png", clustersMap);
  cv::Mat lineMap(5, 5, CV_8U, cv::Scalar_<uint8_t>(0));
  lineMap.at<uint8_t>(0, 0) = -1;
  lineMap.at<uint8_t>(1, 0) = -1;
  lineMap.at<uint8_t>(2, 0) = -1;
  lineMap.at<uint8_t>(3, 0) = -1;
  lineMap.at<uint8_t>(0, 2) = -1;
  lineMap.at<uint8_t>(1, 2) = -1;
  lineMap.at<uint8_t>(2, 2) = -1;
  lineMap.at<uint8_t>(3, 2) = -1;
  lineMap.at<uint8_t>(3, 3) = -1;
  lineMap.at<uint8_t>(3, 4) = -1;
  LOG(INFO) << '\n' << lineMap << '\n';
  int N = zxm::SegmentByLines(clustersMap, lineMap);
  LOG(INFO) << "Number of Classes: " << N << '\n';
  zxm::tool::DrawClusters("../dbg/testSegmentByLines1.png", clustersMap);
}


//PASS法向量聚类方法
void testDetectPlanes() {
  //1,18,55,99
  cv::Mat colorImg =
    zxm::tool::CV_Imread1920x1440(R"(..\vendor\ELSED\images\55_scene.png)", cv::IMREAD_COLOR, cv::INTER_NEAREST);
  cv::Mat mask =
    cvDNN::blobFromNPY(R"(E:\VS_Projects\MonoPlanner\example\data\55_mask.npy)", CV_8U);
  cv::Mat normalMap =
    cvDNN::blobFromNPY(R"(E:\VS_Projects\MonoPlanner\example\data\55_normal.npy)", CV_32F);
  normalMap = zxm::tool::CV_Convert32FTo32FC3(normalMap, mask);
  zxm::tool::DrawNormals("../dbg/inNormals.png", normalMap);
  //
  cv::Mat planesMat = zxm::DetectPlanes(colorImg, normalMap, true);
  //
  LOG(INFO) << "nClass = " << zxm::tool::cvMax<int32_t>(planesMat) + 1 << '\n';
  zxm::tool::DrawClusters("../dbg/outPlanesMap.png", planesMat);
}


int main(int argc, char* argv[])
{
  try {
    testDetectPlanes();
  } catch (const std::exception &e) {
    LOG(INFO) << e.what();
  } catch (...) {
    LOG(INFO) << "Unknown Error.";
  }
  return 0;
}