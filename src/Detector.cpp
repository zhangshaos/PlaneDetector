//
// Created by zxm on 2022/12/5.
//

#include <set>
#include <map>
#include <vector>
#include <opencv2/core.hpp>
#include <glog/logging.h>

#include "../HyperConfig.h"
#include "Detector.h"
#include "Tools.h"
#include "DetectLine.h"


cv::Mat zxm::DetectPlanes(const cv::Mat &colorImg,
                          const cv::Mat &normalMap,
                          bool enableDebug) {
  cv::Mat clusterMap;
  zxm::ClusteringByNormal(clusterMap, normalMap, enableDebug);
  cv::Mat clusterColorMap;
  if (enableDebug)
    clusterColorMap = zxm::tool::DrawClusters("../dbg/NormalClusters.png", clusterMap);
  auto lineMap = zxm::CreateStructureLinesMap(colorImg, normalMap, true, enableDebug);
  if (enableDebug)
    zxm::tool::DrawClusters("../dbg/LinesMap.png", lineMap);
  if (enableDebug) {
    const int Rows = clusterColorMap.size[0], Cols = clusterColorMap.size[1];
    CV_Assert(Rows == lineMap.size[0] && Cols == lineMap.size[1]);
    for (int i = 0; i < Rows; ++i)
      for (int j = 0; j < Cols; ++j)
        if (lineMap.at<int32_t>(i, j) > 0)
          clusterColorMap.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
    zxm::tool::CV_ImWriteWithPath("../dbg/BlendNormalClustersAndLines.png", clusterColorMap);
  }
  zxm::SegmentByLines(clusterMap, lineMap, enableDebug);
  return clusterMap;
}


int zxm::ClusteringByNormal(cv::Mat &clusterMap,
                            const cv::Mat &normalMap,
                            bool enableDebug) {
  using Pxi32_t = std::pair<int, int>;
  CV_Assert(normalMap.type() == CV_32FC3);

  const int Rows = normalMap.size[0], Cols = normalMap.size[1];
  auto isValidIndex = [&Rows, &Cols](int y, int x) -> bool {
    return 0 <= y && y < Rows && 0 <= x && x < Cols;
  };
  auto isContinuedAngle = [](const cv::Vec3f &v0, const cv::Vec3f &v1) -> bool {
    return acos(zxm::tool::clamp(v0.dot(v1), -1.f, 1.f)) <= (TH_CONTINUED_ANGLE * CV_PI / 180);
  };
  auto isDiscreted = [](const cv::Vec3f &v0, const cv::Vec3f &v1) -> bool {
    return acos(zxm::tool::clamp(v0.dot(v1), -1.f, 1.f)) >= (2 * TH_CONTINUED_ANGLE * CV_PI / 180);
  };
  // first pass
  zxm::tool::ClassUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      const cv::Vec3f &v2 = normalMap.at<cv::Vec3f>(i, j);
      std::map<int, cv::Vec3f> connectedCls;//???????????????????????????????????????????????????????????????
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, -1},
                                  Pxi32_t{-1, 0},
                                  Pxi32_t{-1, 1}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (isValidIndex(targetY, targetX)) {
          bool isContinued = false;
          const cv::Vec3f &v1 = normalMap.at<cv::Vec3f>(targetY, targetX);
          /*
          const cv::Vec3f v0 =
            isValidIndex(targetY+yxOff.first, targetX+yxOff.second) && !isOnLine(targetY,targetX) ?
            normalMap.at<cv::Vec3f>(targetY + yxOff.first, targetX + yxOff.second) : v1;
          */
          /*const cv::Vec3f &v2 = normalMap.at<cv::Vec3f>(i, j);*/
          const cv::Vec3f v3 = isValidIndex(i - yxOff.first, j - yxOff.second) ?
                               normalMap.at<cv::Vec3f>(i - yxOff.first, j - yxOff.second) : v2;
          //
          isContinued = (isContinuedAngle(v1, v2) &&
                         isContinuedAngle(v2, v3) &&
                         isContinuedAngle(v1, v3)) ||//????????????????????????????????????
                        (isContinuedAngle(v1, v2) &&
                         isDiscreted(v2, v3));//??????????????????????????????????????????
          zxm::tool::CheckMathError();
          if (isContinued) {
            int32_t C = resultCls.at<int32_t>(targetY, targetX);
            if (C < 0)
              throw std::logic_error("Processed label of (targetY,targetX) should >= 0!"
                                     " in zxm::ClusteringByNormal()");
            connectedCls.emplace(C, v1);
            if (C < minConnectedCls)
              minConnectedCls = C;
          }
        }
      }//??????(i,j)???????????????????????????
      if (connectedCls.empty()) {
        //?????????
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      } else {
        if (connectedCls.size() > 1) {
          const auto &normal0 = connectedCls.at(minConnectedCls);
          for (const auto &[C, normal1] : connectedCls) {
            if (C != minConnectedCls && isContinuedAngle(normal0, normal1))
              //????????????????????????????????????
              mergeCls.unionClass(minConnectedCls, C);
          }
        }
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
    }//one pass over.
  }

  if (enableDebug)
    zxm::tool::DrawClusters("../dbg/NormalClustersBeforeMerge.png", resultCls);

  std::vector<uint32_t> shrinkClass;
  int nCls = (int) mergeCls.shrink(&shrinkClass);
  // second pass
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      int32_t C = resultCls.at<int32_t>(i, j);
      C = (int32_t) shrinkClass[C];
      resultCls.at<int32_t>(i, j) = C;
    }
  }
  // resultCls...
  swap(resultCls, clusterMap);
  return nCls;
}


// ??????zxm::raster()??????
void TestRaster() {
  auto l1 = zxm::raster(17, 121, 1, 1),
    l2 = zxm::raster(120, 77, 23, 23),
    l3 = zxm::raster(100, 1, 97, 33),
    l4 = zxm::raster(97, 11, 37, 29),
    l5 = zxm::raster(100, 34, 100, 100),
    l6 = zxm::raster(77, 120, 11, 120);
  LOG(INFO) << l1.front().y << ' ' << l1.front().x << ' ' << l1.back().y << ' ' << l1.back().x
            << " Should be " << "1 1 17 121" << '\n';
  LOG(INFO) << l2.front().y << ' ' << l2.front().x << ' ' << l2.back().y << ' ' << l2.back().x
            << " Should be " << "23 23 120 77" << '\n';
  LOG(INFO) << l3.front().y << ' ' << l3.front().x << ' ' << l3.back().y << ' ' << l3.back().x
            << " Should be " << "100 1 97 33" << '\n';
  LOG(INFO) << l4.front().y << ' ' << l4.front().x << ' ' << l4.back().y << ' ' << l4.back().x
            << " Should be " << "97 11 37 29" << '\n';
  LOG(INFO) << l5.front().y << ' ' << l5.front().x << ' ' << l5.back().y << ' ' << l5.back().x
            << " Should be " << "100 34 100 100" << '\n';
  LOG(INFO) << l6.front().y << ' ' << l6.front().x << ' ' << l6.back().y << ' ' << l6.back().x
            << " Should be " << "11 120 77 120" << '\n';
  LOG(INFO) << std::endl;
  auto vs = l1;
  vs.insert(vs.end(), l2.begin(), l2.end());
  vs.insert(vs.end(), l3.begin(), l3.end());
  vs.insert(vs.end(), l4.begin(), l4.end());
  vs.insert(vs.end(), l5.begin(), l5.end());
  vs.insert(vs.end(), l6.begin(), l6.end());
  cv::Mat result(192, 256, CV_8U, cv::Scalar_<uint8_t>(0));
  for (const auto v: vs) {
    result.at<uint8_t>(v.y, v.x) = 0xff;
  }
  zxm::tool::CV_ImWriteWithPath("../dbg/raster.png", result);
}


cv::Mat zxm::CreateStructureLinesMap(const cv::Mat &colorImg,
                                     const cv::Mat &normalMap,
                                     bool extend,
                                     bool enableDebug) {
  CV_Assert(colorImg.type() == CV_8UC3);
  CV_Assert(normalMap.type() == CV_32FC3);
  CV_Assert(colorImg.size[0] >= normalMap.size[0]);
  CV_Assert(colorImg.size[1] >= normalMap.size[1]);
  std::vector<std::vector<cv::Point2i>> lines;
#ifdef USE_EDLib
  zxm::EDLinesWrapper edlines(colorImg);
  lines = edlines.detect();
#endif
#ifdef USE_ELSED
  zxm::ELSEDWrapper elsed;
  lines = elsed.detect(colorImg);
#endif
  //??????????????????????????????????????????????????????
  std::sort(lines.begin(), lines.end(), [](const std::vector<cv::Point2i> &l,
                                           const std::vector<cv::Point2i> &r)
                                           { return l.size()>r.size(); });
  //???????????????????????????normalMap??????
  int Rows = colorImg.size[0], Cols = colorImg.size[1];
  float scaleY = 1.f, scaleX = 1.f;
  if (Rows > normalMap.size[0] && Cols > normalMap.size[1]) {
    scaleY = (float) normalMap.size[0] / Rows,
    scaleX = (float) normalMap.size[1] / Cols;
    Rows = normalMap.size[0];
    Cols = normalMap.size[1];
  }
  //??????????????????????????????????????????????????????or?????????????????????????????????????????????????????????????????????????????????
  auto isDiffSide = [&normalMap](int iy0, int ix0, int iy1, int ix1) {
    const auto
      &v0 = normalMap.at<cv::Vec3f>(iy0, ix0),
      &v1 = normalMap.at<cv::Vec3f>(iy1, ix1);
    const float angle = acos(zxm::tool::clamp(v0.dot(v1), -1.f, 1.f));
    zxm::tool::CheckMathError();
    return angle > float(TH_DIFF_SIDE_ANGLE * CV_PI / 180);
  };
  //??????????????????????????????lines
  cv::Mat edgeResult = enableDebug ?
                       cv::Mat(Rows, Cols, CV_8U, cv::Scalar_<uint8_t>(0)) :
                       cv::Mat{};
  //??????????????????result
  cv::Mat result(Rows, Cols, CV_32S, cv::Scalar_<int32_t>(-1));
  int32_t startID = 1;
  std::vector<int32_t> linesResultID(lines.size(), -1);//?????????????????????result??????ID
  for (size_t i=0,iEnd=lines.size(); i<iEnd; ++i) {
    auto &l = lines[i];
    if (l.size() < 2)
      continue;
    //????????????????????????????????????????????????(y0,x0)???(y1,x1)
    const float
      deltaY = float(l.back().y - l.front().y),
      deltaX = float(l.back().x - l.front().x);
    const float
      dy = (deltaY / sqrt(deltaY * deltaY + deltaX * deltaX)) * GAP_HALF_DIFF_SIDE,
      dx = (deltaX / sqrt(deltaY * deltaY + deltaX * deltaX)) * GAP_HALF_DIFF_SIDE;
    zxm::tool::CheckMathError();
    const float
      y0 = -dx, x0 = dy,
      y1 = dx, x1 = -dy;//(dy,dx) ?????????????????????????????????90??
    //?????????e??????????????????????????????
    size_t nDiffSide = 0;
    for (auto &px : l) {
      LOG_ASSERT(0 <= px.y && px.y < colorImg.size[0] &&
                 0 <= px.x && px.x < colorImg.size[1]);
      //???????????????????????????????????????
      px.y = int((px.y + 0.5f) * scaleY);
      px.x = int((px.x + 0.5f) * scaleX);
      int iy0 = int(px.y + 0.5f + y0),
        ix0 = int(px.x + 0.5f + x0),
        iy1 = int(px.y + 0.5f + y1),
        ix1 = int(px.x + 0.5f + x1);
      if (0 <= iy0 && iy0 < Rows && 0 <= ix0 && ix0 < Cols &&
          0 <= iy1 && iy1 < Rows && 0 <= ix1 && ix1 < Cols &&
          isDiffSide(iy0, ix0, iy1, ix1))
        ++nDiffSide;
    }
    if (nDiffSide > std::max(size_t(STRUCTURE_LINE_RATIO * l.size()), size_t(1))) {
      //????????????e???????????????????????????????????????????????????e????????????????????????????????????
      for (const auto &px : l) {
        auto &pxID = result.at<int32_t>(px.y, px.x);
        if (pxID < 0)
          pxID = startID;
      }
      linesResultID[i] = startID;
      ++startID;
    }
    //
    if (enableDebug)
      for (const auto &px : l)
        edgeResult.at<uint8_t>(px.y, px.x) = 0xff;
  }
  if (enableDebug) {
    zxm::tool::CV_ImWriteWithPath("../dbg/RawEdges.png", edgeResult);
    zxm::tool::DrawClusters("../dbg/LinesMapBeforeExtend.png", result);
  }
  if (!extend)
    return result;
  //
  auto isValid = [&Rows, &Cols](const cv::Point2i &px) -> bool {
    return 0 <= px.y && px.y < Rows && 0 <= px.x && px.x < Cols;
  };
  auto getLineID = [&result](const cv::Point2i &px) -> int32_t {
    return result.at<int32_t>(px.y, px.x);
  };
  //????????????result????????????
  for (size_t i=0,iEnd=lines.size(); i<iEnd; ++i) {
    if (linesResultID[i] < 0)
      continue;
    const auto &l = lines[i];
    const auto &startPx = l.front(), &endPx = l.back();
    //LOG_ASSERT(linesResultID[i]==getLineID(startPx));
    //LOG_ASSERT(linesResultID[i]==getLineID(endPx));//result????????????????????????
    //?????????????????????????????????????????????????????????or????????????l????????????
    //?????????????????????????????????????????????????????????????????????
    //?????????????????????????????????????????????
    float
    deltaY = float(endPx.y - startPx.y),
    deltaX = float(endPx.x - startPx.x);
    cv::Point2i startStartPx(int(startPx.x + 0.5f - deltaX),
                             int(startPx.y + 0.5f - deltaY));
    cv::Point2i endEndPx(int(endPx.x + 0.5f + deltaX),
                         int(endPx.y + 0.5f + deltaY));
    //???????????????
    do {
      auto exLine1 = zxm::raster(startPx.y, startPx.x, startStartPx.y, startStartPx.x);
      int endIdx = 0;
      bool hit = false;
      for (const auto &px: exLine1) {
        if (!isValid(px))
          break;
        if (getLineID(px) >= 0 && getLineID(px) != linesResultID[i]) {
          hit = true;
          break;
        }
        ++endIdx;
      }
      if (!hit)
        break;
      for (size_t j = 0; j < endIdx; ++j) {
        int iy = exLine1[j].y, ix = exLine1[j].x;
        result.at<int32_t>(iy, ix) = linesResultID[i];
      }
    } while (0);
    //???????????????
    do {
      auto exLine2 = zxm::raster(endPx.y, endPx.x, endEndPx.y, endEndPx.x);
      int endIdx = 0;
      bool hit = false;
      for (const auto &px: exLine2) {
        if (!isValid(px))
          break;
        if (getLineID(px) >= 0 && getLineID(px) != linesResultID[i]) {
          hit = true;
          break;
        }
        ++endIdx;
      }
      if (!hit)
        break;
      for (size_t j = 0; j < endIdx; ++j) {
        int iy = exLine2[j].y, ix = exLine2[j].x;
        result.at<int32_t>(iy, ix) = linesResultID[i];
      }
    } while (0);
  }
  return result;
}


int zxm::SegmentByLines(cv::Mat &clusterMap,
                        const cv::Mat &lineMap,
                        bool enableDebug) {
  using Pxi32_t = std::pair<int, int>;
  CV_Assert(clusterMap.type() == CV_32S);
  CV_Assert(lineMap.type() == CV_32S);
  CV_Assert(clusterMap.size == lineMap.size);

  const int Rows = clusterMap.size[0], Cols = clusterMap.size[1];
  // first pass
  zxm::tool::ClassUnion mergeCls;
  cv::Mat resultCls(Rows, Cols, CV_32S, -1);
  int nextCls = 0;
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      int32_t cCur = clusterMap.at<int32_t>(i, j);
      int32_t curLine = lineMap.at<int32_t>(i, j);
      if (curLine > 0)
        //??????????????????????????????????????????
        continue;
      //??????????????????(i,j)?????????4????????????????????????
      std::set<int> connectedCls;//???????????????????????????????????????????????????????????????
      int minConnectedCls = std::numeric_limits<int>::max();
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, 0}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols) {
          int32_t cTarget = clusterMap.at<int32_t>(targetY, targetX);
          int32_t targetLine = lineMap.at<int32_t>(targetY, targetX);
          //?????????????????????????????????????????????????????????
          //1???c???t?????????????????? ??????ID??????????????????????????????connectedCls???maxConnectedCls
          //2???c???????????????t?????? ?????????
          if (targetLine <= 0 && (cTarget == cCur)) {
            int32_t C = resultCls.at<int32_t>(targetY, targetX);
            if (C < 0)
              throw std::logic_error("Processed label of (targetY,targetX) should >= 0!"
                                     " in zxm::SegmentByLines()");
            connectedCls.emplace(C);
            if (C < minConnectedCls)
              minConnectedCls = C;
          }
        }
      }//for all adjacent pixel
      if (connectedCls.empty()) {
        //?????????
        mergeCls.tryInsertClass(nextCls);
        resultCls.at<int32_t>(i, j) = nextCls++;
      } else {
        if (connectedCls.size() > 1) {
          for (int C : connectedCls) {
            if (C != minConnectedCls)
              mergeCls.unionClass(minConnectedCls, C);
          }
        }
        resultCls.at<int32_t>(i, j) = minConnectedCls;
      }
    }//one pass for all pixels
  }
  //???????????????????????????????????????????????????
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      int32_t cCur = clusterMap.at<int32_t>(i, j);
      int32_t curLine = lineMap.at<int32_t>(i, j);
      if (curLine <= 0)
        //???????????????????????????????????????
        continue;
      //1??????????????????????????????4????????????cID????????????????????????????????????????????????
      //2????????????1???????????????????????????????????????cID????????????????????????????????????????????????
      //3????????????2?????????????????????????????????cID????????????????????????????????????????????????????????????ID???
      //4????????????cID????????????????????????ID???
      bool found = false;
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, 0},
                                  Pxi32_t{0, 1},
                                  Pxi32_t{1, 0}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols) {
          int32_t cTarget = clusterMap.at<int32_t>(targetY, targetX);
          int32_t targetLine = lineMap.at<int32_t>(targetY, targetX);
          if (cTarget == cCur && targetLine <= 0) {
            resultCls.at<int32_t>(i, j) = resultCls.at<int32_t>(targetY, targetX);
            found = true;//break do-while
            break;//break for
          }
        }
      }
      if (found)
        continue;//go to next line pixel
      for (const Pxi32_t &yxOff: {Pxi32_t{0, -1},
                                  Pxi32_t{-1, 0}}) {
        int targetY = i + yxOff.first, targetX = j + yxOff.second;
        if (0 <= targetY && targetY < Rows &&
            0 <= targetX && targetX < Cols) {
          int32_t cTarget = clusterMap.at<int32_t>(targetY, targetX);
          int32_t targetLine = lineMap.at<int32_t>(targetY, targetX);
          if (cTarget == cCur && targetLine > 0) {
            resultCls.at<int32_t>(i, j) = resultCls.at<int32_t>(targetY, targetX);
            found = true;//break do-while
            break;//break for
          }
        }
      }
      if (found)
        continue;//go to next line pixel
      //??????????????????ID
      mergeCls.tryInsertClass(nextCls);
      resultCls.at<int32_t>(i, j) = nextCls++;
    }//one pass for line pixel
  }

  if (enableDebug)
    zxm::tool::DrawClusters("../dbg/ClustersBeforeMerge.png", resultCls);

  std::vector<uint32_t> shrinkClass;
  int nCls = (int) mergeCls.shrink(&shrinkClass);
  // second pass
  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Cols; ++j) {
      int32_t C = resultCls.at<int32_t>(i, j);
      if (C < 0)
        throw std::logic_error("Processed label should >= 0!"
                               " in zxm::SegmentByLines()");
      resultCls.at<int32_t>(i, j) = (int) shrinkClass[C];
    }
  }
  swap(clusterMap, resultCls);
  return nCls;
}
