#include "opencv2/objdetect.hpp"   //36007bytes      26
#include "opencv2/highgui.hpp"     //36215  动态链接库24
#include "opencv2/imgproc.hpp"     //228169          24
#include <iostream>
//所用动态链接库  107916
//haarcascade_frontalface_alt2.xml   540616
//可执行文件 42216 bytess
using namespace std;
using namespace cv;
class Face_detect{
private:
    bool tryflip;
    CascadeClassifier cascade;
    double scale;
    string cascadeName;
public:
    Face_detect(string casfile1,int scale1=1,bool tryflip1=true){
        cascadeName =casfile1;
        scale =scale1;
        tryflip =tryflip1;
    }
    int init(){
        if (!cascade.load(samples::findFile(cascadeName)))
        {
            cerr << "ERROR: Could not load classifier cascade" << endl;
            help();
            return 0;
        }
        return 1;
    }
    void help()
        {
             cout << "\nThis program demonstrates the use of cv::CascadeClassifier class to detect objects (Face + eyes). You can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [--try-flip]\n"
               "   [filename|camera_index]\n\n"
            "see facedetect.cmd for one call:\n"
            "./facedetect --cascade=\"../haarcascade_frontalface_alt.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
        }
        int detectFaces(Mat& img)
        {
             double t = 0;
            vector<Rect> faces;
            const static Scalar colors[] =
            {
                Scalar(255,0,0),
                Scalar(255,128,0),
                Scalar(255,255,0),
                Scalar(0,255,0),
                Scalar(0,128,255),
                Scalar(0,255,255),
                Scalar(0,0,255),
                Scalar(255,0,255)
            };
            Mat gray, smallImg;

            cvtColor( img, gray, COLOR_BGR2GRAY );
            double fx = 1 / scale;
            resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
            equalizeHist( smallImg, smallImg ); //直方图均衡化，，用于提高图像的质量

            t = (double)getTickCount();
            cascade.detectMultiScale( smallImg, faces,
               1.1, 2, 0
              //|CASCADE_FIND_BIGGEST_OBJECT
              //|CASCADE_DO_ROUGH_SEARCH
              |CASCADE_SCALE_IMAGE,
              Size(30, 30) );
             t = (double)getTickCount() - t;
            printf( "detection time = %g ms\n", t*1000/getTickFrequency());
            int facesSize=faces.size();
            if(facesSize>0)
                cout<<"检测到人脸数目为：\t"<<facesSize<<endl;
            else cout<<"w未检测到行人"<<endl;
            return facesSize;
        }
        void detectAndDraw( Mat& img )
        {
            double t = 0;
            vector<Rect> faces;
            const static Scalar colors[] =
            {
                Scalar(255,0,0),
                Scalar(255,128,0),
                Scalar(255,255,0),
                Scalar(0,255,0),
                Scalar(0,128,255),
                Scalar(0,255,255),
                Scalar(0,0,255),
                Scalar(255,0,255)
            };
            Mat gray, smallImg;

            cvtColor( img, gray, COLOR_BGR2GRAY );
            double fx = 1 / scale;
            cout<<"scale\t"<<scale<<" \t fx \t "<<fx<<"\tINTER_LINEAR_EXACT:\t"<<INTER_LINEAR_EXACT<<endl;
            if(gray.empty ()) {
                cout<<"gray frame is empty,ERROR happened"<<endl;
                cout<<"gray.size\t"<<gray.depth ()<<endl;
                //return -1;
            }else{
                cout<<"gray frame isnot empty,go on"<<endl;
            }
            resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT );
            equalizeHist( smallImg, smallImg ); //直方图均衡化，，用于提高图像的质量

            t = (double)getTickCount();
            cascade.detectMultiScale( smallImg, faces,
               1.1, 2, 0
              //|CASCADE_FIND_BIGGEST_OBJECT
              //|CASCADE_DO_ROUGH_SEARCH
              |CASCADE_SCALE_IMAGE,
              Size(30, 30) );
             t = (double)getTickCount() - t;
            printf( "detection time = %g ms\n", t*1000/getTickFrequency());
            int count=faces.size();
            cout<<"检测到人脸\t"<<count<<"\t 个人脸"<<endl;
            for ( size_t i = 0; i < count; i++ )
            {
                cout<<"正在处理第\t"<<i<<"\t 个人脸"<<endl;
                Rect r = faces[i];
                Mat smallImgROI;
                Point center;
                Scalar color = colors[i%8];
                int radius;

                double aspect_ratio = (double)r.width/r.height;
                rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
                putText(img, "people detected",         //输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
                Point(r.x*scale, r.y),
                FONT_HERSHEY_COMPLEX,1, // font face and scale
                Scalar(220, 20, 60), // 
                2, LINE_AA); // line thickness and type   
            }
            imshow( "result", img );
    }
};

int main( int argc, const char** argv )
{
    VideoCapture capture;
    Mat frame, image;
    string casfile1="../haarcascade_frontalface_alt2.xml";
    string casfile2="../haarcascade_frontalface_default.xml";
    string casfile3="../haarcascade_frontalface_alt_tree.xml";
    int camera =0;
    Face_detect *detect=new Face_detect(casfile1,1,true);
    if(!detect->init())
    {
        cout<<"Face_detect ERROR,can't load the cv::CascadeClassifier file"<<casfile1<<endl;
        return -1;
    }
    if(!capture.open(camera))
    {
        cout << "Capture from camera #" <<  camera << " didn't work" << endl;
        return 1;
    }
    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;

        for(;;)
        {
            capture >> frame;
            if( frame.empty() )
                break;

            Mat frame1 = frame.clone();
            detect->detectAndDraw( frame1);

            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }
    return 0;
}


