//
//  treesDetection.h
//

#ifndef treesDetection_h
#define treesDetection_h


#include <iostream>

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"


class treesDetection
{
public:
    // basic constructor
    treesDetection(void);
    // function that load all the test images
    std::vector<cv::Mat> loadTestImages(std::string path);
    // function that returns all the thresholded proposal regions
    std::vector<cv::Rect> obtainProposalRegions(cv::Mat img, cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> search);
    // function that classifies all the proposal regions and return only the good ones
    std::vector<cv::Rect> regionsClassification(cv::Mat img, std::vector<cv::Rect> goodRects, cv::dnn::Net cnn_net);
    // function data makes a green bounding box around each detected trees
    void detectionBoundingBox(cv::Mat img, std::vector<cv::Rect> classified);
};

#endif /* treesDetection_h */
