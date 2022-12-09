//
// Created by zxm on 2022/12/9.
//

#ifndef PLANEDETECTOR_PLANEDETECTOR_H
#define PLANEDETECTOR_PLANEDETECTOR_H


template<typename LineDetector>
class PlaneDetector {
public:
  PlaneDetector();



private:
  LineDetector lineDetector_;
};


#endif //PLANEDETECTOR_PLANEDETECTOR_H
