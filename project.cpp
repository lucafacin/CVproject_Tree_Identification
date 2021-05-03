//
//  Luca Facin 1197485
//  Computer Vision 2019/2020
//
//  Final Project
//  09/07/2020
//

#include "opencv2/ximgproc/segmentation.hpp"
#include <opencv2/dnn.hpp>

#include "treesDetection.cpp"

int main(int argc, char** argv)
{
    // creating an object of class Selective Search Segmentation
    cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> search = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
    // importing a freeze version CNN model for tree classification trained in Google Colab
    cv::dnn::Net cnn_net = cv::dnn::readNetFromTensorflow("../trained_cnn.pb");
    // creating an object of class treesDetection
    treesDetection detector = treesDetection();
    // loading all the test images
    std::string path = "../data/*.jpg";
    std::vector<cv::Mat> images = detector.loadTestImages(path);
    // cycling for all the test images
    for(int i=0; i < images.size(); i++)
    {
        // obtaing only the thresholded proposal segments
        std::vector<cv::Rect> goodSegment = detector.obtainProposalRegions(images[i], search);
        // classyfing the proposal segments obtaining only the good ones
        std::vector<cv::Rect> goodClassified = detector.regionsClassification(images[i], goodSegment, cnn_net);
        // making a green bounding box around each detected trees
        detector.detectionBoundingBox(images[i], goodClassified);
        cv::destroyAllWindows();
    }
    
    return 0;
}
