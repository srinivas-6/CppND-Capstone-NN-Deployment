#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;


class Input
{
public:
    
    
    Input(){}
    ~Input(){}
    
    void display(string kWindow,Mat input_image);
    void loadStream(CommandLineParser parser, VideoCapture &cap, VideoWriter &video);
    void saveStream(CommandLineParser parser, Mat frame, VideoWriter &video);
    

    
private:
    
    Mat input_image;
    string outputFile;
    
    

    
};