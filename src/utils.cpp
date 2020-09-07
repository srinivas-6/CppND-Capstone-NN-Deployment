#include "utils.h"
#include <vector>

using namespace cv;
using namespace std;
using namespace dnn;


Util::Util(const Util &other)
{
    confThreshold = other.confThreshold;
    nmsThreshold = other.nmsThreshold;
}

void Util::loadClasses()
{
    
    ifstream ifs(classesFile.c_str());
    string get_line;
    while (getline(ifs, get_line)) 
        classes.push_back(get_line);

    
}

/* reference taken from source https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/ */

// Remove the bounding boxes with low confidence using non-maxima suppression
void Util::postprocess(Mat &frame, const vector<Mat> &outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        Util::drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}


// Draw the predicted bounding box
void Util::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<string> Util::getOutputsNames(const Net &net)
{
    static vector<string> output_names;
    if (output_names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        output_names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        output_names[i] = layersNames[outLayers[i] - 1];
    }
    return output_names;
}

YOLO::YOLO(const YOLO &other):Util(other.confThreshold, other.nmsThreshold)
{
    modelConfig = other.modelConfig;
    modelWeights = other.modelWeights;
}

YOLO::YOLO(const YOLO &&other):Util(std::move(other.confThreshold), std::move(other.nmsThreshold))
{
    modelConfig = std::move(other.modelConfig);
    modelWeights = std::move(other.modelWeights);
}

YOLO &YOLO::operator=(const YOLO &other)
{
    if(this==&other)
        return *this;
    
    modelConfig = other.modelConfig;
    modelWeights = other.modelWeights;
    Util(other.confThreshold, other.nmsThreshold);

    return *this;

}

YOLO &YOLO::operator=(const YOLO &&other)
{
    if(this== &other)
        return *this;
    
    modelConfig = std::move(other.modelConfig);
    modelWeights = std::move(other.modelWeights);
    Util(std::move(other.confThreshold), std::move(other.nmsThreshold));
    return *this;
}

Mat YOLO::Detect(Mat &frame)
{
    Mat blob;
    vector <Mat> outs;
   
    blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        
        //Sets the input to the network
    net.setInput(blob);
    
    net.forward(outs, getOutputsNames(net));
    postprocess(frame,outs);
    FPS(frame);

    return std::move(frame);
}

void YOLO::FPS(Mat &frame)
{
        // Get inference time
    vector<double> layersTimes;
    double freq = getTickFrequency();
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("FPS : %.2f", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
    
}
