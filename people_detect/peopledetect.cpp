#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

class Detector
{
    enum Mode { Default, Daimler } m;
    HOGDescriptor hog, hog_d;
public:
    Detector() : m(Default), hog(), hog_d(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9)
    {
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        hog_d.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    }
    void toggleMode() { m = (m == Default ? Daimler : Default); }
    string modeName() const { return (m == Default ? "Default" : "Daimler"); }
    vector<Rect> detect(InputArray img)
    {
        // Run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        vector<Rect> found;
        if (m == Default)
            hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2, false);
        else if (m == Daimler)
            hog_d.detectMultiScale(img, found, 0.5, Size(8,8), Size(32,32), 1.05, 2, true);
        return found;
    }
    void adjustRect(Rect & r) const
    {
        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
    }
    void drawText(Mat & image,Rect & r)
    {
         putText(image, "people detected",         //输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
         Point(r.x, r.y),
         FONT_HERSHEY_COMPLEX, 0.5, // font face and scale
         Scalar(220, 20, 60), // 
         2, LINE_AA); // line thickness and type
    }
};

static const string keys = "{ help h   |   | print help message }"
                           "{ camera c | 0 | capture video from camera (device index starting from 0) }"
                           "{ video v  |   | use video as input }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("This sample demonstrates the use ot the HoG descriptor.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    int camera = parser.get<int>("camera");
    string file = parser.get<string>("video");
    cout<<"the pareser result  camera "<<camera<<"\nvideo : "<<file<<endl;
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }

    VideoCapture cap;
    if (file.empty()){
        cout<<"begin to opent the camera/n"<<endl;
        cap.open(camera);
    }
    else
    {
        cout<<"begin to open the file: "<<file<<endl;
        file = samples::findFileOrKeep(file);
        cap.open(file);
    }
    if (!cap.isOpened())
    {
        cout << "Can not open video stream: '" << (file.empty() ? "<camera>" : file) << "'" << endl;
        return 2;
    }else{
        cout<<"open the video stream successful"<<endl;
    }

    cout << "Press 'q' or <ESC> to quit." << endl;
    cout << "Press <space> to toggle between Default and Daimler detector" << endl;
    Detector detector;
    Mat frame;
    for (;;)
    {
        cout<<"begin to get a frame"<<endl;
        cap >> frame;
        if (frame.empty())
        {
            cout << "Finished reading: empty frame" << endl;
            break;
        }
        //cout<<"begin to show a frame"<<endl;
        //imshow("People", frame);
        int64 t = getTickCount();
        vector<Rect> found = detector.detect(frame);
        t = getTickCount() - t;

        // show the window
        {
            ostringstream buf;
            buf << "Mode: " << detector.modeName() << " ||| "
                << "FPS: " << fixed << setprecision(1) << (getTickFrequency() / (double)t);
            putText(frame, buf.str(), Point(10, 30), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 2, LINE_AA);
        }
        for (vector<Rect>::iterator i = found.begin(); i != found.end(); ++i)
        {
            Rect &r = *i;
            detector.adjustRect(r);
            rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2); //输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
            detector.drawText(frame,r);
        }
        imshow("People detector", frame);

        // interact with user
        const char key = (char)waitKey(30);
        if (key == 27 || key == 'q') // ESC
        {
            cout << "Exit requested" << endl;
            break;
        }
        else if (key == ' ')
        {
            detector.toggleMode();
        }
    }
    return 0;
}
