#include "treesDetection.h"

// basic constructor
treesDetection::treesDetection(void)
{
    return;
};


// function that load all the test images
std::vector<cv::Mat> treesDetection::loadTestImages(std::string path)
{
    std::vector<cv::Mat> images;
    std::vector<cv::String> files;
    cv::glob(path, files);
    for (int i = 0; i < files.size(); i++)
    {
        cv::Mat image = cv::imread(files[i]);
        // resizing image for speed-up the segmentation algorithm
        int newHeight = 200;
        int newWidth = image.cols * newHeight / image.rows;
        cv::resize(image, image, cv::Size(newWidth, newHeight));
        // saving all the images in a vector
        images.push_back(image);
    }
    
    // return the vector with all the test images
    return images;
};


// function that returns all the thresholded proposal regions
std::vector<cv::Rect> treesDetection::obtainProposalRegions(cv::Mat img, cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> search)
{
    // initialization
    std::vector<cv::Rect> rects, goodRects;
    // setting the image on which running segmentation
    search->setBaseImage(img);
    // type of selective search (Fast reduce the number of proposal region)
    search->switchToSelectiveSearchFast();
    // preforming the segmentation and saving all the segment in a rect
    search->process(rects);
    // different thresholds variables
    int r = -20, h = 60, w = 50;
    // cycling for all the proposal segments
    for(int j = 0; j < rects.size(); j++)
    {
        // threshold on aspect ratio, height and width
        if (rects[j].height - rects[j].width >= r & rects[j].height > h & rects[j].width > w)
        {
            // saving only the thresholded proposal regions
            goodRects.push_back(rects[j]);
        }
    }
    
    // return only the thresholded proposal regions
    return goodRects;
};

    
// function that classify all the proposal regions and return only the good ones
std::vector<cv::Rect> treesDetection::regionsClassification(cv::Mat img, std::vector<cv::Rect> goodRects, cv::dnn::Net cnn_net)
{
    // output image to show the proposal regions
    cv::Mat proposalImg = img.clone();
    // initialization of vector containing only good classified proposal regions
    std::vector<cv::Rect> classified;
    // for all the proposal regions
    for(int j = 0; j < goodRects.size(); j++)
    {
        // print a red rectangle
        cv::rectangle(proposalImg, goodRects[j], cv::Scalar(0, 0, 255));
        // obtain the relative segment of the image
        cv::Mat segImage = img(goodRects[j]);
        // reshape the segment image in order to CNN layout
        cv::Mat testSegment = cv::dnn::blobFromImage(segImage, 255.0, cv::Size(32, 32), cv::Scalar(0), true);
        // set the segment image in the first layer of the network
        cnn_net.setInput(testSegment);
        // make the classification
        cv::Mat test = cnn_net.forward();
        // keeping only the good classified proposal regions
        if(test.at<float>(0, 0) == 1)
        {
            classified.push_back(goodRects[j]);
        }
    }
    
    // showing all the proposal region passed to the classifier
    cv::imshow("Proposal Regions", proposalImg);
    cv::waitKey(0);
    
    // retrun only the good classified segments
    return classified;
};


// function data makes a green bounding box around each detected trees
void treesDetection::detectionBoundingBox(cv::Mat img, std::vector<cv::Rect> classified)
{
    // initialization of different tree regions
    std::vector<cv::Rect> firstTree, secondTree;
    // output image with correct classified trees
    cv::Mat imageOut = img.clone();
    // if there are good classified region
    // copare the relative position to understand if there are more than one tree
    if(!classified.empty())
    {
        // obtain the first good region
        cv::Rect base = classified[0];
        // for all the good classified region
        for(int j = 0; j < classified.size(); j++)
        {
            // threshold distance between good classified region (empirically found)
            int dist = 150;
            // if the relative distance of the regions centers is under distance treshold first tree
            if(std::abs((classified[j].x + classified[j].width/2) - (base.x + base.width/2)) < dist & std::abs((classified[j].y + classified[j].width/2) - (base.y + base.width/2)) < dist)
            {
                firstTree.push_back(classified[j]);
            }
            // if the relative distance of the regions centers is over distance treshold second tree
            else
            {
                secondTree.push_back(classified[j]);
            }
        }
    }
    
    // for all the good classified region around the first tree obtain only one rectagle
    // rely on the mean value of height, with and coordinate of top left corner
    if(!firstTree.empty())
    {
        // initialization of parameter
        int height = 0, width = 0, x = 0, y = 0, count = 0;
        // evaluation of mean value
        for(int k = 0; k < firstTree.size(); k++)
        {
            height += firstTree[k].height;
            width += firstTree[k].width;
            x = x + firstTree[k].x;
            y = y + firstTree[k].y;
            count = count + 1;
        }
        x = x / count;
        y = y / count;
        height = height / count;
        width = width / count;
        // showing only one good green rect around the tree
        cv::Rect bb (x, y, width, height);
        cv::rectangle(imageOut, bb, cv::Scalar(0, 255, 0));
    }
    
    // for all the good classified region around the second tree obtain only one rectagle
    // rely on the mean value of height, with and coordinate of top left corner
    if(!secondTree.empty())
    {
        // initialization of parameter
        int height = 0, width = 0, x = 0, y = 0, count = 0;
        // evaluation of mean value
        for(int k = 0; k < secondTree.size(); k++)
        {
            height += secondTree[k].height;
            width += secondTree[k].width;
            x = x + secondTree[k].x;
            y = y + secondTree[k].y;
            count = count + 1;
        }
        x = x / count;
        y = y / count;
        height = height / count;
        width = width / count;
        // showing only one good green rect around the tree
        cv::Rect bb (x, y, width, height);
        cv::rectangle(imageOut, bb, cv::Scalar(0, 255, 0));
    }
    
    // showing the final image with reactangle around each tree
    cv::imshow("Detected Trees", imageOut);
    cv::waitKey(0);
    
};
