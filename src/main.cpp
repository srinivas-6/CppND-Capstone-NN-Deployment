
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <future>

#include "input.h"
#include "utils.h"


using namespace std;
using namespace cv;
using namespace dnn;


const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./yolo --image= path to image file --conf=confThreshold --nms=nmsThreshold \n\t\t./yolo --video=path to video file -- --conf=confThreshold --nms=nmsThreshold}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
"{conf c        |<none>| confidence Threshold}"
"{nms n        |<none>| Non-max suppression Threshold}";

float confThreshold = 0.0 ; // Confidence threshold
float nmsThreshold = 0.0 ; // Non-max supression threshold

int main(int argc, char **argv) 
{
    
    Mat _frame; // frame to hold input stream
    
    // Open video or image stream
    VideoCapture cap;
    // Write output stream
    VideoWriter video;

    CommandLineParser parser(argc, argv,keys);

    //Path to YOLO weights and architecture and postprocessing parameters from Cmdline parser

    string modelConfiguration = "/home/workspace/CppND-Capstone-NN-Deployment/yolo_model/yolov3.cfg";
    string modelWeights = "/home/workspace/CppND-Capstone-NN-Deployment/yolo_model/yolov3.weights";

    if(parser.has("conf"))
      confThreshold = parser.get<float>("c");
    
    else
    {
      confThreshold = 0.5; // default value of confidence threshold
    }
    
    if(parser.has("nms"))
      nmsThreshold = parser.get<float>("n");
    
    else
    {
      nmsThreshold = 0.5; // default value of non-max suppression
    }

    std::unique_ptr<YOLO> model = std::make_unique<YOLO>(modelConfiguration, modelWeights, confThreshold, nmsThreshold);
    

    std::unique_ptr<Input> _stream = std::make_unique<Input>();  
   
    while(waitKey(1)<0)
    {
        
    _stream->loadStream(parser,cap,video);

    cap >> _frame;

    if(_frame.empty())
    {
      cout << "Processing Done !!!" << "\n";
      waitKey(3000);
     
    }
     
    Mat frame = model->Detect(_frame);
    

    // Convert frame to CV_8U and write the frame with the detection boxes
    Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);

    static const string kWindow = "Yolov3 ObjectDetection in C++";

   // Display detected results
   _stream->display(kWindow,frame);

   // Save image / video stream
   _stream->saveStream(parser,frame, video);

  }

  cap.release();

  return 0;
  
}
