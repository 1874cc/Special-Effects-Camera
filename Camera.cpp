#include "Camera.h"
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include<iostream>
#include<windows.h>
#include<qtimer.h>
#include<qtimezone.h>
#include <opencv2\objdetect.hpp>
using namespace std;
using namespace cv;
Camera::Camera(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
	initPhoto();
	time = new QTimer(this);
	openCamera(0);//打开摄像头
	setDelay(1000 / getFrameRate() * 1.0);
	connect(time, &QTimer::timeout, this, &Camera::run);
}

bool Camera::openCamera(int device)
{
	f_number = 0;
	//释放资源，防止已有资源与VideoCapture实例关联
	capture.release();
	TAKE_PHOTO = false;
	//打开视频文件
	
	if (capture.open(device)) {
		ui.label->setText(QString::fromLocal8Bit("打开摄像头成功"));
		return true;
	}
	else {
		ui.viewLabel->setText(QString::fromLocal8Bit("未找到摄像头"));
		return false;
	}

	//return capture.open(device);
}

double Camera::getFrameRate()
{
	double rate = capture.get(CV_CAP_PROP_FPS);
    return rate;
}

void Camera::stopIt()
{
	stop = true;
}

bool Camera::isStopped()
{
	return stop;
}

QString Camera::getNowTime()
{
	QTime qtime = QTime::currentTime();
	QDate qyear = QDate::currentDate();//获取本地时间作为照片文件名，保证照片唯一性
	QString text = qyear.toString("yyyyMMdd-")+qtime.toString("HHmmss");
	return text;
}
bool Camera::readNextFrame(cv::Mat& frame)
{
	return capture.read(frame);
}
void Camera::setDelay(int d)
{
	delay = d;
	time->start(delay);
}
void Camera::initPhoto()//初始化处理字key
{
	TAKE_PHOTO = false;
	GLASS_SHOW = false;
	OLD_SHOW = false;
	SOBEL_SHOW = false;
	GAUSSIANBLUR_SHOW = false;
	HSV_SHOW = false;
	SHARPEN_SHOW = false;
	RAIN_SHOW = false;
	FACERECOGNITION_SHOW = false;
	SKETCH_SHOW = false;
	CONTRASTPLUS_SHOW= false;
}

void Camera::run() {
	cv::Mat frame;//存储当前帧
	cv::Mat output;//存储输出帧
	stop = false;
	while (!isStopped()) {
		if (!readNextFrame(frame)) {
			break;
		}

		output = frame;//输出帧等于输入帧
		f_number++;
		////在此对图像进行处理↓↓↓↓↓↓↓↓↓↓↓↓↓↓==================↓↓↓↓↓↓↓↓↓↓↓↓↓↓
		if (GLASS_SHOW)//如果毛玻璃选中	
			output = glassShow(output);
		if (OLD_SHOW)//复古风
			output = oldShow(output);
		if (SOBEL_SHOW)//浮雕
			output = sobelShow(output);
		if (GAUSSIANBLUR_SHOW)//磨皮（高斯模糊
			output = gaussianBlurShow(output);
		if (HSV_SHOW)//绘画效果
			output=hsvShow(output);
		if (SHARPEN_SHOW)//锐化
			output = sharpenShow(output);
		if (RAIN_SHOW)//下雨
			output = rainShow(output);
		if (FACERECOGNITION_SHOW)//人脸检测
			output = faceRecognitionShow(output);
		if (SKETCH_SHOW)//漫画素描
			output = sketchShow(output);
		if (CONTRASTPLUS_SHOW)//对比度+
			output = contrastPlusShow(output);
		////在此对图像进行处理↑↑↑↑↑↑↑↑↑↑↑↑↑↑==================↑↑↑↑↑↑↑↑↑↑↑↑↑↑	
		
		//在qt界面上显示处理后输出的图像
		cv::cvtColor(output, output, CV_BGR2RGB);
		qImg = QImage((const unsigned char*)(output.data), output.cols, output.rows, output.step, QImage::Format_RGB888);
		ui.viewLabel->setPixmap(QPixmap::fromImage(qImg));
		ui.viewLabel->show();
		ui.label->setText(QString::fromLocal8Bit("加载摄像头图像成功"));

		//如果按下了拍照按钮，则保存这一帧图片
		if (TAKE_PHOTO == true) {
			ui.label->setText(QString::fromLocal8Bit("正在拍照"));
			QString savePath = getNowTime();
			qImg.save(savePath + ".jpg", "JPG", 100);
			TAKE_PHOTO = false;
		}

		return;
	}
}
//---------------------------0毛玻璃------------------------------------------------------//
cv::Mat Camera::glassShow(cv::Mat& image)
{
	int width = image.cols;
	int heigh = image.rows;
	RNG rng;
	Mat img1(image.size(), CV_8UC3);
	for (int y = 1; y < heigh - 1; y++)
	{
		uchar* P0 = image.ptr<uchar>(y);
		uchar* P1 = img1.ptr<uchar>(y);
		for (int x = 1; x < width - 1; x++)
		{
			int tmp = rng.uniform(0, 9);
			P1[3 * x] = image.at<uchar>(y - 1 + tmp / 3, 3 * (x - 1 + tmp % 3));
			P1[3 * x + 1] = image.at<uchar>(y - 1 + tmp / 3, 3 * (x - 1 + tmp % 3) + 1);
			P1[3 * x + 2] = image.at<uchar>(y - 1 + tmp / 3, 3 * (x - 1 + tmp % 3) + 2);
		}
	}
	return img1;
}

//---------------------------1复古风------------------------------------------------------//
cv::Mat Camera::oldShow(cv::Mat& image)//复古风图像效果style
{
	int width = image.cols;
	int heigh = image.rows;
	RNG rng;
	Mat img1(image.size(), CV_8UC3);
	for (int y = 0; y < heigh; y++)
	{
		uchar* P0 = image.ptr<uchar>(y);
		uchar* P1 = img1.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			float B = P0[3 * x];
			float G = P0[3 * x + 1];
			float R = P0[3 * x + 2];
			float newB = 0.272 * R + 0.534 * G + 0.131 * B;
			float newG = 0.349 * R + 0.686 * G + 0.168 * B;
			float newR = 0.393 * R + 0.769 * G + 0.189 * B;
			if (newB < 0)newB = 0;
			if (newB > 255)newB = 255;
			if (newG < 0)newG = 0;
			if (newG > 255)newG = 255;
			if (newR < 0)newR = 0;
			if (newR > 255)newR = 255;
			P1[3 * x] = (uchar)newB;
			P1[3 * x + 1] = (uchar)newG;
			P1[3 * x + 2] = (uchar)newR;
		}
	}
	return img1;
}
//---------------------------2浮雕------------------------------------------------------//
cv::Mat Camera::sobelShow(cv::Mat& image)//浮雕图像效果
{
	Mat img0(image.size(), CV_8UC3);
	Mat img1(image.size(), CV_8UC3);
	for (int y = 1; y < image.rows - 1; y++)
	{
		uchar* p0 = image.ptr<uchar>(y);
		uchar* p1 = image.ptr<uchar>(y + 1);

		uchar* q0 = img0.ptr<uchar>(y);
		uchar* q1 = img1.ptr<uchar>(y);

		for (int x = 1; x < image.cols - 1; x++)
		{
			for (int i = 0; i < 3; i++)
			{
				int tmp0 = p1[3 * (x + 1) + i] - p0[3 * (x - 1) + i] + 128;//浮雕
				if (tmp0 < 0)
					q0[3 * x + i] = 0;
				else if (tmp0 > 255)
					q0[3 * x + i] = 255;
				else
					q0[3 * x + i] = tmp0;

				int tmp1 = p0[3 * (x - 1) + i] - p1[3 * (x + 1) + i] + 128;//雕刻
				if (tmp1 < 0)
					q1[3 * x + i] = 0;
				else if (tmp1 > 255)
					q1[3 * x + i] = 255;
				else
					q1[3 * x + i] = tmp1;
			}
		}
	}
	return img0;
}
//---------------------------3磨皮（高斯模糊）------------------------------------------------------//
cv::Mat Camera::gaussianBlurShow(cv::Mat& image)
{
	cv::Mat gaussianImage = image.clone();
	Mat dst_step1;//高美颜后的图
	Mat dst;//高斯模糊图
	int value1 = 3, value2 = 1; //磨皮程度与细节程度的确定
	int dx = value1 * 5;    //双边滤波参数之一  
	double fc = value1 * 12.5; //双边滤波参数之一  
	int p = 50; //透明度  
	Mat temp1, temp2, temp3, temp4;

	//双边滤波  
	bilateralFilter(gaussianImage, temp1, dx, fc, fc);
	temp2 = (temp1 - gaussianImage + 128);

	//高斯模糊  
	GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
	temp4 = image + 2 * temp3 - 255;
	dst = (image * (100 - p) + temp4 * p) / 100;
	dst.copyTo(dst_step1);
	return dst;
}
//---------------------------4 hsv水彩绘画------------------------------------------------------//
cv::Mat Camera::hsvShow(cv::Mat& image) {
	//转换成HSV色彩空间
	cv::Mat hsv;
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
	//把三个通道分隔进三个图像中
	vector<cv::Mat> channels;
	cv::split(hsv, channels);
	////channels[0]是色调（h分量）
	////channels[1]是饱和度（S分量）
	////channels[2]是亮度（V分量）

	//用色调，饱和度与亮度表示颜色---》改变亮度为固定值
	channels[2] = 255;
	//重新合并通道
	cv::merge(channels, hsv);
	//转换回RGB
	cv::Mat RGB_image;
	cv::cvtColor(hsv, RGB_image, cv::COLOR_HSV2BGR);
	return RGB_image;
}
//---------------------------5锐化------------------------------------------------------//
cv::Mat Camera::sharpenShow(cv::Mat& image) {
	//判断是否需要分配图像数据。如果需要，就分配
	//cv::Mat result;
	//result.create(image.size(), image.type());
	//int nchannels = image.channels();//获得通道数
	////处理所有行（出了第一行和最后一行）
	//for (int j = 1; j < image.rows - 1; j++) {
	//	const uchar* previous = image.ptr<const uchar>(j - 1);//分配指针
	//	const uchar* current = image.ptr<const uchar>(j);
	//	const uchar* next = image.ptr<const uchar>(j + 1);

	//	uchar* output = result.ptr<uchar>(j);
	//	for (int i = nchannels; i < (image.cols - 1) * nchannels; i++) {
	//		//应用锐化算子
	//		*output++ = cv::saturate_cast<uchar>(
	//			5 * current[i] - current[i - nchannels] -
	//			current[i + nchannels] - previous[i] - next[i]
	//			);
	//	}
	//}
	//result.row(0).setTo(cv::Scalar(0));
	//result.row(result.rows - 1).setTo(cv::Scalar(0));
	//result.col(0).setTo(cv::Scalar(0));
	//result.col(result.cols - 1).setTo(cv::Scalar(0));
	//return result;

	//新方法
	//构造内核，所有入口都初始化为0
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	//对内核赋值
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	//对图像滤波
	cv::Mat result;
	cv::filter2D(image, result, image.depth(), kernel);
	return result;
}
//---------------------------6下雨------------------------------------------------------//
cv::Mat Camera::rainShow(cv::Mat& image) {
	cv::Mat rain = cv::imread("rain.png");
	cv::Mat result;
	cv::Mat src = image.clone();
	cv::resize(rain, rain, cv::Size(src.cols, src.rows), CV_INTER_LINEAR);
	cv::add(src * 0.9, rain * 0.4, result);
	return result;
}
//---------------------------7人脸检测------------------------------------------------------//
cv::Mat Camera::faceRecognitionShow(cv::Mat& image) {
	if (!face_cascade.load("haarcascade_frontalface_alt2.xml")) {//加载人脸检测数据
		ui.label_tip->setText(QString::fromLocal8Bit("人脸检测训练数据未找到！"));
		return image;
	}
	cv::Mat output_faces = image;
	int nSize = 30;//人脸框最小尺寸
	vector<cv::Rect> faces;
	//执行多尺度人脸检测
	face_cascade.detectMultiScale(
		image,//输入图
		faces,//用于保存人脸矩形的vector
		1.1,//多尺度检测所采用的递增缩放系数
		3,//用于控制误检的参数，至少3次重叠检测才认为人脸确实存在
		cv::CASCADE_DO_CANNY_PRUNING,//利用canny算子在检测中排除边缘少的区域
		cv::Size(nSize, nSize)//最小人脸尺寸
	);
	if (faces.size() > 0) {
		//如果检测到人脸
		//在output显示出所有的人脸矩形
		for (int i = 0; i < faces.size(); i++) {
			cv::rectangle(output_faces, faces[i], cv::Scalar(0, 0, 255));
		}
		faces.clear();//清空保存人脸矩形的vector
	}
	return output_faces;
}
//---------------------------8素描------------------------------------------------------//
cv::Mat Camera::sketchShow(cv::Mat& image)
{
	//转灰度图(一般针对灰度图进行Sobel滤波)
	cv::Mat gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	//水平方向Sobel滤波
	cv::Mat sobelX;
	cv::Sobel(gray,	// 输入图像
		sobelX,	// 输出图像
		CV_8U,	// 输出图像像素深度
		1, 0,		// x和y方向导数阶数
		3,		// 正方形内核尺寸，必须为奇数
		0.4, 128);	// 缩放比例和偏移量(默认值为1和0)

	//垂直方向滤波
	cv::Mat sobelY;
	cv::Sobel(gray,	// 输入图像
		sobelY,	// 输出图像
		CV_8U,	// 输出图像像素深度
		0, 1,		// x和y方向导数阶数
		3,		// 正方形内核尺寸，必须为奇数
		0.4, 128);	// 缩放比例和偏移量(默认值为1和0)
	// 计算Sobel滤波器的模
	cv::Sobel(gray, sobelX, CV_16S, 1, 0);
	cv::Sobel(gray, sobelY, CV_16S, 0, 1);
	cv::Mat sobel;
	// 计算L1模
	sobel = abs(sobelX) + abs(sobelY);
	// 计算Sobel算子，必须用浮点数类型
	cv::Sobel(gray, sobelX, CV_32F, 1, 0);
	cv::Sobel(gray, sobelY, CV_32F, 0, 1);
	// 计算梯度的L2模和方向
	cv::Mat norm, dir;
	cv::cartToPolar(sobelX, sobelY, norm, dir); 	// 笛卡尔坐标转极坐标

	//为了更清晰地以图像形式显示Sobel滤波器的模，将其转为灰度图，并将最大模值对应为黑色：
		// 找到Sobel中像素最大值
	double sobmin, sobmax;
	cv::minMaxLoc(sobel, &sobmin, &sobmax);
	// 转换成8位图像
	// sobelImage = -255. / sobmax * sobel + 255
	cv::Mat sobelImage;
	sobel.convertTo(sobelImage, CV_8U,
		-255. / sobmax, 255);
	cv::cvtColor(sobelImage, sobelImage, CV_GRAY2BGR);		
	return sobelImage;
}
//---------------------------9对比度+------------------------------------------------------//
cv::Mat Camera::contrastPlusShow(cv::Mat& image)
{
	cv::Mat result;
	cv::Mat hsv;
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);//将rgb转为hsv空间
	vector<cv::Mat>channels3;
	cv::split(hsv, channels3);//拆分为三个通道，channels3[2]为亮度通道
	cv::equalizeHist(channels3[2], channels3[2]);//进行opencv自带的直方图均衡化操作
	cv::merge(channels3, hsv);//合并通道
	cv::cvtColor(hsv, result, cv::COLOR_HSV2BGR);//HSV空间转换回BGR
	return result;
}

//=============复选框选效果======================
void Camera::isGlassCheck()//当毛玻璃选择中
{
	if(ui.checkBox_galssshow->isChecked()) {
		GLASS_SHOW = true;
	}
	else {
		GLASS_SHOW = false;
	}
}
void Camera::isOldCheck()//复古风
{
	if (ui.checkBox_oldstyle->isChecked())
		OLD_SHOW = true;
	else
		OLD_SHOW = false;
}
void Camera::isSobelCheck()//浮雕
{
	if (ui.checkBox_sobel->isChecked())
		SOBEL_SHOW = true;
	else
		SOBEL_SHOW = false;
}
void Camera::isGaussianBlurCheck()//磨皮
{
	if (ui.checkBox_gaussianBlur->isChecked())
		GAUSSIANBLUR_SHOW = true;
	else
		GAUSSIANBLUR_SHOW = false;
}
void Camera::isHsvCheck() {//绘画
	if (ui.checkBox_hsv->isChecked())
		HSV_SHOW = true;
	else
		HSV_SHOW = false;
}
void Camera::isSharpenCheck() {//锐化
	if (ui.checkBox_sharpen->isChecked())
		SHARPEN_SHOW = true;
	else {
		SHARPEN_SHOW = false;
	}
}
void Camera::isRainCheck() {//下雨
	if (ui.checkBox_rain->isChecked())
		RAIN_SHOW = true;
	else {
		RAIN_SHOW = false;
	}
}
void Camera::isFaceRecognitionCheck() {//人脸检测
	if (ui.checkBox_faceRecognition->isChecked()) {
		FACERECOGNITION_SHOW = true;
	}
	else
		FACERECOGNITION_SHOW = false;
}
void Camera::isSketchCheck()//素描
{
	if (ui.checkBox_sketch->isChecked())
		SKETCH_SHOW = true;
	else
		SKETCH_SHOW = false;
}
void Camera::isContrastPlusCheck()//对比度增强
{
	if (ui.checkBox_contrastPlus->isChecked())
		CONTRASTPLUS_SHOW = true;
	else
		CONTRASTPLUS_SHOW = false;
}
//=============复选框选效果======================
//=============拍照按钮======================
void Camera::saveImg() {
	TAKE_PHOTO = true;
}
