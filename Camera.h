#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_Camera.h"
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include<iostream>
#include <opencv2\objdetect.hpp>
using namespace std;
class Camera : public QMainWindow
{
    Q_OBJECT

public:
    Camera(QWidget *parent = Q_NULLPTR);
    bool openCamera(int device);//打开摄像头
    double getFrameRate();//获取视频的帧速率：
    void stopIt();//定义成员方法用于终止处理过程以及返回终止状态：//结束处理
    bool isStopped();//处理过程是否已经终止？
    QString getNowTime();//获取本地时间作为输出的图像文件名
    bool readNextFrame(cv::Mat& frame);//定义成员方法读取下一帧//设置延时ms,0表示每一帧都等待，负数表示不延时
    void setDelay(int d);//cout << "当前处理帧率：" << 1000.0/d << endl;
    //图片处理函数
    void initPhoto();//初始化图片处理
    cv::Mat glassShow(cv::Mat& image);//毛玻璃
    cv::Mat oldShow(cv::Mat& image);//复古风
    cv::Mat sobelShow(cv::Mat& image);//浮雕效果
    cv::Mat gaussianBlurShow(cv::Mat& image);//磨皮（高斯模糊
    cv::Mat hsvShow(cv::Mat& image);//hsv空间绘画效果
    cv::Mat sharpenShow(cv::Mat& image);//锐化效果
    cv::Mat rainShow(cv::Mat& image);//下雨效果
    cv::Mat faceRecognitionShow(cv::Mat& imgge);//人脸识别
    cv::Mat sketchShow(cv::Mat& image);//素描
    cv::Mat contrastPlusShow(cv::Mat& image);//刮风
public slots:
    void saveImg();//拍照按钮的点击事件，保存图片到本地
    void run();//核心处理函数，抓取并处理视频中的图像
    //检测复选框状态
    void isGlassCheck();
    void isOldCheck(); 
    void isSobelCheck();
    void isGaussianBlurCheck();
    void isHsvCheck();
    void isSharpenCheck();
    void isRainCheck();
    void isFaceRecognitionCheck();
    void isSketchCheck();
    void isContrastPlusCheck();
private:
    Ui::CameraClass ui;
    QImage qImg;//输出图像（用于显示与保存到本地
    cv::Mat outputImg;//处理中的图像，获取摄像头图像，并进行处理
    cv::VideoCapture capture; //opencv视频捕获类对象!!!
    cv::CascadeClassifier face_cascade;//用于人脸检测的级联分类器
    long f_number;//已经处理的帧数
    double delay;//帧之间的延时
    QTimer *qTime;//帧之间的延时
    bool stop;//结束处理表直
    QTimer* time;
    bool TAKE_PHOTO;//拍照判别字
    bool GLASS_SHOW;//效果判别字
    bool OLD_SHOW;
    bool SOBEL_SHOW;
    bool GAUSSIANBLUR_SHOW;
    bool HSV_SHOW;
    bool SHARPEN_SHOW;
    bool RAIN_SHOW;
    bool FACERECOGNITION_SHOW;
    bool SKETCH_SHOW;
    bool CONTRASTPLUS_SHOW;
};
