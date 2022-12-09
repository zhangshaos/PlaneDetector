// 针对 ELSED 和 EDLines 算法进行包装
// Created by zxm on 2022/12/9.
//

#ifndef PLANEDETECTOR_DETECTLINE_H
#define PLANEDETECTOR_DETECTLINE_H

#include <memory>
#ifdef USE_EDLib
#include <EDColor.h>
#include <EDLines.h>
#endif
#ifdef USE_ELSED
#include <ELSED.h>
#endif
#include <opencv2/core.hpp>


namespace zxm {

#ifdef USE_EDLib
class EDLinesWrapper {
public:
  explicit EDLinesWrapper(const cv::Mat &colorImg);
  std::vector<std::vector<cv::Point2i>>
  detect();

private:
  std::shared_ptr<::EDLines> detector_{nullptr};
};
#endif


#ifdef USE_ELSED
class ELSEDWrapper {
public:
  ELSEDWrapper();
  std::vector<std::vector<cv::Point2i>>
  detect(const cv::Mat &img);

private:
  std::shared_ptr<upm::ELSED> detector_{nullptr};
};
#endif


// 当斜率0<k<1是的中值画线法(ix0,iy0)->(ix1,iy1)
std::vector<cv::Point>
raster_core(int iy0, int ix0, int iy1, int ix1);

// 中值划线算法
std::vector<cv::Point>
raster(int iy0, int ix0, int iy1, int ix1);


}


#endif //PLANEDETECTOR_DETECTLINE_H
