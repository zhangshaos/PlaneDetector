//
// Created by zxm on 2022/12/9.
//
#include "DetectLine.h"
#include <cfenv>
#include <execution>
#include "Tools.h"


#ifdef USE_EDLib
zxm::EDLinesWrapper::EDLinesWrapper(const cv::Mat &colorImg) {
  ::EDColor EdColor(colorImg, 20, 4, 1.2, true);
  auto self = std::make_shared<::EDLines>(EdColor);
  std::swap(self, detector_);
}


std::vector<std::vector<cv::Point2i>>
zxm::EDLinesWrapper::detect() {
  std::vector<std::vector<cv::Point2i>> result;
  auto lines = detector_->getLines();
  for (const auto& l : lines) {
    const int
    iy0 = int(l.start.y + 0.5),
    ix0 = int(l.start.x + 0.5),
    iy1 = int(l.end.y + 0.5),
    ix1 = int(l.end.x + 0.5);
    auto lPx = zxm::raster(iy0, ix0, iy1, ix1);
    result.emplace_back(std::move(lPx));
  }
  return result;
}
#endif


#ifdef USE_ELSED
zxm::ELSEDWrapper::ELSEDWrapper() {
  upm::ELSEDParams params;
  params.anchorThreshold = 4;
  params.gradientThreshold = 20;
  params.ksize = 3;
  params.sigma = 1.2;
  params.minLineLen = 9;
  //params.lineFitErrThreshold = 0.5;
  params.pxToSegmentDistTh = 2.0;
  //params.junctionAngleTh = 20 * M_PI / 180;
  //params.junctionEigenvalsTh = 5;
  params.listJunctionSizes = {5, 7, 9};
  auto self = std::make_shared<upm::ELSED>(params);
  std::swap(self, detector_);
}


std::vector<std::vector<cv::Point2i>>
zxm::ELSEDWrapper::detect(const cv::Mat &img) {
  std::vector<std::vector<cv::Point2i>> result;
  auto lines = detector_->detect(img);
  const int64_t
  Cols = detector_->getImgInfoPtr()->imageWidth,
  Rows = detector_->getImgInfoPtr()->imageHeight;
  //fixme: ELSED float error.
  std::feclearexcept(FE_ALL_EXCEPT);
  for (const auto &l : lines) {//x0, y0, x1, y1
    const int
    ix0 = (int)l(0),
    iy0 = (int)l(1),
    ix1 = (int)l(2),
    iy1 = (int)l(3);
    assert(0<=iy0 && iy0<Rows && 0<=ix0 && ix0<Cols);
    assert(0<=iy1 && iy1<Rows && 0<=ix1 && ix1<Cols);
    auto lPx = zxm::raster(iy0, ix0, iy1, ix1);
    result.emplace_back(std::move(lPx));
  }
  return result;
}
#endif


// 当斜率0<k<1是的中值画线法(ix0,iy0)->(ix1,iy1)
std::vector<cv::Point>
zxm::raster_core(int iy0, int ix0, int iy1, int ix1) {
  std::vector<cv::Point> result;
  double k = double(iy1 - iy0) / (ix1 - ix0);
  double b = (iy1 + 0.5) - k * (ix1 + 0.5);
  result.emplace_back(ix0, iy0);
  int y = iy0;
  for (int x = ix0; x < ix1; ++x) {
    double midX = x + 1.5, midY = y + 1.0;
    double d = k * midX + b - midY;
    if (d >= 0)
      ++y;
    result.emplace_back(x + 1, y);
  }//over
  return result;
}

// 中值划线算法
std::vector<cv::Point>
zxm::raster(int iy0, int ix0, int iy1, int ix1) {
  using namespace std;
  bool inverse = false;
  if (ix0 > ix1) {
    //保证(ix0,iy0)是在x-y坐标系更靠左侧
    swap(ix0, ix1);
    swap(iy0, iy1);
    inverse = true;
  }
  std::vector<cv::Point> result;
  if (iy1 == iy0) {
    //横线
    for (int i = ix0; i <= ix1; ++i)
      result.emplace_back(i, iy0);
  } else if (ix1 == ix0) {
    //纵线
    if (iy0 > iy1) {
      swap(iy0, iy1);//保证iy0在底端
      inverse = true;
    }
    for (int i = iy0; i <= iy1; ++i)
      result.emplace_back(ix0, i);
  } else {
    if (iy1 - iy0 == ix1 - ix0) {
      // k==1
      for (int i = ix0; i <= ix1; ++i)
        result.emplace_back(i, iy0 + (i - ix0));
    } else if (iy1 - iy0 == ix0 - ix1) {
      // k==-1
      for (int i = ix0; i <= ix1; ++i)
        result.emplace_back(i, iy0 - (i - ix0));
    } else {
      //只处理直线斜率k属于(0,1)之间的情况
      double k = double(iy1 - iy0) / (ix1 - ix0);
      if (0 < k && k < 1)
        result = raster_core(iy0, ix0, iy1, ix1);
      else if (k > 1) {
        //交换xy轴
        result = raster_core(ix0, iy0, ix1, iy1);
        std::for_each(std::execution::seq,
                      result.begin(),
                      result.end(),
                      [](cv::Point &px) {
                        swap(px.x, px.y);
                      });
      } else if (k > -1 && k < 0) {
        //x轴翻转，y -> -y
        result = raster_core(-iy0, ix0, -iy1, ix1);
        std::for_each(std::execution::seq,
                      result.begin(),
                      result.end(),
                      [](cv::Point &px) {
                        px.y *= -1;
                      });
      } else if (k < -1) {
        //x轴翻转，然后交换xy轴
        result = raster_core(ix0, -iy0, ix1, -iy1);
        std::for_each(std::execution::seq,
                      result.begin(),
                      result.end(),
                      [](cv::Point &px) {
                        swap(px.x, px.y);
                        px.y *= -1;
                      });
      } else
        throw std::logic_error("zxm::raster() Impossible!");
    }
  }
  if (inverse) {
    std::vector<cv::Point> invResult(result.rbegin(), result.rend());
    result.swap(invResult);
  }
  return result;
}