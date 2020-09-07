#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

class Util
{
    public:
    
    Util(float conf, float nms):confThreshold(conf),nmsThreshold(nms){}
    ~Util(){}
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame);
    void postprocess(Mat &frame, const vector<Mat> &out);
    vector<string> getOutputsNames(const Net &net);
    void loadClasses();
    Util(const Util &other);

    protected:
    
    // Initialize the parameters
    float confThreshold; // Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    int inpWidth = 416;  // Width of network's input image
    int inpHeight = 416; // Height of network's input image
    vector<string> classes;
    string classesFile = "yolo_model/coco.names";

};


class YOLO : public Util
{
    public:
    
    YOLO(string config, string weights, float conf, float nms):Util{conf,nms}, modelConfig(config), modelWeights(weights)
    { 
        net = readNetFromDarknet(modelConfig, modelWeights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        cout << "Loaded YOLOv3 model with Backend : " << "DNN_BACKEND_OPENCV " << "\n"; 

        loadClasses();
    }

    ~YOLO(){}
    YOLO(const YOLO &other);
    YOLO &operator=(const YOLO &other);
    Mat Detect(Mat &frame);
    YOLO(const YOLO &&other);
    YOLO &operator=(const YOLO &&other);


    private:
    
    string modelConfig;
    string modelWeights;
    Net net;
    float conf, nms;

    void FPS(Mat &frame);
    
}; 

