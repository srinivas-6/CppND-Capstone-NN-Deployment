#include "input.h"


void Input::loadStream(CommandLineParser parser, VideoCapture &cap, VideoWriter &video)
{

    parser.about("C++ based object detection using YOLO3 Deep Learning Model");
    if (parser.has("help"))
    {
        parser.printMessage();
       
    }

    string str;   
    
    try {
        
        outputFile = "yolo_out_cpp.avi";
        if (parser.has("image"))
        {
            // Open the image file
            str = parser.get<string>("image");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
            outputFile = str;
        }
        else if (parser.has("video"))
        {
            // Open the video file
            str = parser.get<String>("video");
            ifstream ifile(str);
            if (!ifile) throw("error");
            cap.open(str);
            str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
            outputFile = str;
        }
        // Open the webcam
        else cap.open(parser.get<int>("device"));
        
    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
    
    }
    
    // Get the video writer initialized to save the output video
    if (!parser.has("image")) {
        video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    }

   
}


void Input::display(string kWindow, Mat image)
{
    
    imshow(kWindow, image);
    //waitKey(0);
}

void Input::saveStream(CommandLineParser parser, Mat frame, VideoWriter &video)
{
    if(parser.has("image"))
        imwrite(outputFile,frame);
    else
    {
        video.write(frame);
    }
    
}