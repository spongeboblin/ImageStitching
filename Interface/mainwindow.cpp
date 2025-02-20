#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);
	/*初始化*/
	init();
	/*connect*/
	connect(ui.action,SIGNAL(triggered()),this,SLOT(slotOpenDirectory()));
	connect(this,SIGNAL(signalShowImageList()),this,SLOT(slotShowImageList()));
	/*选定影像，并显示*/
	connect(ui.treeWidget,SIGNAL(itemDoubleClicked(QTreeWidgetItem *, int)),this,SLOT(slotShowSelectedImage(QTreeWidgetItem*, int)));
	/*特征提取，并显示结果*/
	connect(ui.pushButton,SIGNAL(clicked()),this,SLOT(slotExtractFeature()));//特征提取
	connect(this,SIGNAL(signalShowFeaturePoint()),this,SLOT(slotShowFeaturePoint()));
	/*特征匹配，并显示结果*/
	connect(ui.pushButton_2,SIGNAL(clicked()),this,SLOT(slotMatchFeature()));//特征匹配
	connect(this,SIGNAL(signalShowMatching()),this,SLOT(slotShowMatching()));
	/*影像对齐，并显示结果*/
	connect(ui.pushButton_3,SIGNAL(clicked()),this,SLOT(slotAlignImage()));//影像对齐
	connect(this,SIGNAL(signalShowAlignment()),this,SLOT(slotShowAlignment()));

	connect(this,SIGNAL(signalShowConsole()),this,SLOT(slotShowConsole()));

	connect(ui.action_2,SIGNAL(triggered()),this,SLOT(slotViewAlignment()));
	connect(ui.action_3,SIGNAL(triggered()),this,SLOT(slotSave()));
}

MainWindow::~MainWindow()
{

}



void MainWindow::init()
{
	this->showMaximized();
	ui.treeWidget->setHeaderLabel(tr("影像目录"));
	ui.tabWidget->setCurrentIndex(0);//指定tabwedget中的当前tab

	leftImagePath = "";							//左边影像路径
	rightImagePath = "";						//右边影像路径
	leftImageMat = NULL;						//左边影像，矩阵存储
	rightImageMat = NULL;						//右边影像，矩阵存储
	featureType = "";							//特征点类型
	leftImageDesc = NULL;						//左边影像的特征点描述子
	rightImageDesc = NULL;						//右边影像的特征点描述子
	globalHomo = NULL;							//Homography
	matchMat = NULL;							//匹配结果
	xoffset = 0;								//影像映射后x方向上需要的位移
	yoffset = 0;								//影像映射后y方向上需要的位移
	alfa = 0;
	lambda = 0;
	folderPath = "";

	/*初始化操作标志为false*/
	isOpenFlag = false;
	isSelectedImageFlag = false;
	isOverlapFlag = false;
	isExtractedFlag = false;
	isMatchedFlag = false;
	isAlignedFlag = false;
	controlGUI(0);
}



void MainWindow::showImageList(QString path)
{
	//QList<QTreeWidgetItem *> treeWidgetItemList;
	//获取qPath路径下的文件
	QString qPath;
	qPath = path;
	QStringList fileList;
	QDir dir(qPath);
	if (!dir.exists())
	{
		return;
	}
	dir.setFilter(QDir::Dirs|QDir::Files);
	dir.setSorting(QDir::Name | QDir::Reversed);
	QFileInfoList list = dir.entryInfoList();
	//循环
	int i = 0;
	int filecount = 0;
	QTreeWidgetItem *imageItem = new QTreeWidgetItem(ui.treeWidget,QStringList(QString("影像列表")));
	do 
	{
		QFileInfo fileInfo = list.at(i);
		if(fileInfo.fileName() == "." || fileInfo.fileName()== "..") 
		{ 
			//qDebug()<<"filedir="<<fileInfo.fileName();
			i++;
			continue;
		}
		bool bisDir=fileInfo.isDir();
		if (bisDir)
		{
		}
		else
		{
			QString currentFileName = fileInfo.fileName();
			bool Reght=currentFileName.endsWith(".JPG", Qt::CaseInsensitive);
			if(Reght) 
			{ 
 				filecount++;
 				QTreeWidgetItem *imageItem_1 = new QTreeWidgetItem(imageItem,QStringList(currentFileName)); //子节点
				imageItem_1->setIcon(0,QIcon(folderPath+currentFileName));
				imageItem->addChild(imageItem_1); //添加子节点
			} 
			i++;
		}
	} while (i<list.size());
	ui.treeWidget->expandAll(); //结点全部展开
}



void MainWindow::showLeftImage()
{
	Mat mat = leftImageMat.clone();
	QImage qimg = util.Mat2QImage(mat);
	ui.label->clear();
	int label_width = ui.label->width();
	int label_height = ui.label->height();
	int image_width = qimg.width();
	int image_height = qimg.height();
	int scaled_width, scaled_height;
	util.getScaledRatio(image_width,image_height,label_width,label_height,&scaled_width,&scaled_height);

	ui.label->setPixmap(QPixmap::fromImage(qimg).scaled(scaled_width,scaled_height,Qt::KeepAspectRatio));
}


void MainWindow::showRightImage()
{
	Mat mat = rightImageMat.clone();
	QImage qimg = util.Mat2QImage(mat);
	ui.label_2->clear();

	int label_width = ui.label_2->width();
	int label_height = ui.label_2->height();
	int image_width = qimg.width();
	int image_height = qimg.height();
	int scaled_width, scaled_height;
	util.getScaledRatio(image_width,image_height,label_width,label_height,&scaled_width,&scaled_height);

	ui.label_2->setPixmap(QPixmap::fromImage(qimg).scaled(scaled_width,scaled_height,Qt::KeepAspectRatio));
}


void MainWindow::showLeftImageFeature()
{
	Mat warpImageMat = leftImageMat.clone();
	/*把特征点和原影像显示出来*/
	drawKeypoints(warpImageMat,leftImageKeyPoints,warpImageMat,Scalar(0,0,255));//把特征点画在影像上
	QImage qimg = util.Mat2QImage(warpImageMat); //将Mat转换为QImage

	int label_width = ui.label_3->width();
	int label_height = ui.label_3->height();
	int image_width = qimg.width();
	int image_height = qimg.height();
	int scaled_width, scaled_height;
	util.getScaledRatio(image_width,image_height,label_width,label_height,&scaled_width,&scaled_height);

	ui.label_3->clear();
	ui.label_3->setPixmap(QPixmap::fromImage(qimg).scaled(scaled_width,scaled_height,Qt::KeepAspectRatio));
}


void MainWindow::showRightImageFeature()
{
	Mat referImageMat = rightImageMat.clone();
	/*把特征点和原影像显示出来*/
	drawKeypoints(referImageMat,rightImageKeyPoints,referImageMat,Scalar(0,255,0));//把特征点画在影像上
	QImage qimg = util.Mat2QImage(referImageMat); //将Mat转换为QImage

	int label_width = ui.label_4->width();
	int label_height = ui.label_4->height();
	int image_width = qimg.width();
	int image_height = qimg.height();
	int scaled_width, scaled_height;
	util.getScaledRatio(image_width,image_height,label_width,label_height,&scaled_width,&scaled_height);

	ui.label_4->clear();
	ui.label_4->setPixmap(QPixmap::fromImage(qimg).scaled(scaled_width,scaled_height,Qt::KeepAspectRatio));
}


void MainWindow::showMatching()
{
	Mat matchImage = matchMat.clone();//克隆一个匹配结果matchMat副本，用于后续操作
	QImage qimg = util.Mat2QImage(matchImage); 

	int label_width = ui.label_5->width();
	int label_height = ui.label_5->height();
	int image_width = qimg.width();
	int image_height = qimg.height();
	int scaled_width, scaled_height;
	util.getScaledRatio(image_width,image_height,label_width,label_height,&scaled_width,&scaled_height);
	qDebug() << "label_5 " <<scaled_height << scaled_width << endl;
	ui.label_5->clear();
	ui.label_5->setPixmap(QPixmap::fromImage(qimg).scaled(scaled_width,scaled_height,Qt::KeepAspectRatio));
	ui.label_5->setGeometry((label_width-scaled_width)/2,1,scaled_width,label_height);
}


void MainWindow::showAlignment()
{
	Mat alignMat = alignResult.clone();
	QImage qimg = util.Mat2QImage(alignMat);

	int label_width = ui.label_6->width();
	int label_height = ui.label_6->height();
	int image_width = qimg.width();
	int image_height = qimg.height();
	int scaled_width, scaled_height;
	util.getScaledRatio(image_width,image_height,label_width,label_height,&scaled_width,&scaled_height);
	qDebug() << "label_6 " <<scaled_height << scaled_width << endl;
	ui.label_6->clear();
	ui.label_6->setPixmap(QPixmap::fromImage(qimg).scaled(scaled_width,scaled_height,Qt::KeepAspectRatio));
	ui.label_6->setGeometry((label_width-scaled_width)/2,1,scaled_width,label_height);
}

						/************************************************************************/
						/*								槽函数实现							    */
						/************************************************************************/


void MainWindow::slotOpenDirectory()
{
	//弹出文件夹选择窗口，选择后，确定
	QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
		"/home",
		QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
	folderPath = dir+"\\";
	ui.treeWidget->setHeaderLabel(folderPath);
	consoleText = "打开 "+folderPath+" 目录";
	signalShowConsole();
	signalShowImageList();
	isOpenFlag = true;//文件目录已经打开
}



void MainWindow::slotShowImageList()
{
	showImageList(folderPath);
}



void MainWindow::slotShowSelectedImage(QTreeWidgetItem * qTreeWidgetItem, int column)
{
	QTreeWidgetItem *parent = qTreeWidgetItem->parent();
	if (NULL == parent)
	{
		return;
	}
	int col = parent->indexOfChild(qTreeWidgetItem);
	QString qText = qTreeWidgetItem->text(0);
	QString qPath = folderPath+qText;
	consoleText = "选择"+qText;
	if (1 == col % 2)	//左侧
	{
		leftImageMat = imread(util.q2s(qPath));	//读取左边影像，存为mat格式
		showLeftImage();
	}
	else				//右侧
	{
		rightImageMat = imread(util.q2s(qPath));	//读取右边影像，存为mat格式
		showRightImage();
	}
	signalShowConsole();
	ui.tabWidget->setCurrentIndex(0);
	isSelectedImageFlag = true;
	controlGUI(2);
}



void MainWindow::slotExtractFeature()
{
	consoleText = "开始提取特征..";
	//发出显示控制台输出信号
	signalShowConsole();
	int col = ui.comboBox->currentIndex();
	featureType = ui.comboBox->currentText();
	qDebug() << "featureType = " << featureType << endl;
	//提取特征
	imageProcess.leftImageMat = this->leftImageMat;
	imageProcess.rightImageMat = this->rightImageMat;
	imageProcess.processStep = 1;
	imageProcess.featureType = featureType;
	imageProcess.start();
	imageProcess.wait();
	//imageProcess.extractFeature();
	consoleText = "提取特征结束..";
	//发出显示控制台输出信号
	signalShowConsole();

	leftImageKeyPoints = imageProcess.leftImageKeyPoints;
	rightImageKeyPoints = imageProcess.rightImageKeyPoints;

	//发出显示特征信号
	signalShowFeaturePoint();
	consoleText = "显示特征..";
	//发出显示控制台输出信号
	signalShowConsole();
	isExtractedFlag = true;//经过提取特征
	controlGUI(3);
}



void MainWindow::slotMatchFeature()
{
	consoleText = "开始匹配特征..";
	signalShowConsole();
	//匹配特征
	imageProcess.processStep = 2;
	imageProcess.start();
	imageProcess.wait();
	//imageProcess.matchFeature();
	this->matchMat = imageProcess.matchMat;
	consoleText = "匹配特征结束..";
	signalShowConsole();
	//发出显示匹配结果的信号
	signalShowMatching();
	consoleText = "显示匹配结果..";
	signalShowConsole();
	isMatchedFlag = true;//经过匹配特征
	controlGUI(4);
}

void MainWindow::slotAlignImage()
{
	//(对齐)拼接图片
	imageProcess.processStep = 3;
	imageProcess.start();
	imageProcess.wait();
	//imageProcess.alignImage();
	this->alignResult = imageProcess.alignResult;
	//发出显示拼接结果信号
	emit signalShowAlignment();
	isAlignedFlag = true;	//经过对齐影像
	controlGUI(5);
}


void MainWindow::slotShowFeaturePoint()
{
	showLeftImageFeature();
	showRightImageFeature();
	ui.tabWidget->setCurrentIndex(1);
}


void MainWindow::slotShowMatching()
{
	showMatching();
	ui.tabWidget->setCurrentIndex(2);
}


void MainWindow::slotShowAlignment()
{
	showAlignment();
	ui.tabWidget->setCurrentIndex(3);
}


void MainWindow::slotShowConsole()
{
	ui.plainTextEdit->appendPlainText(consoleText);
	
}

void MainWindow::slotViewAlignment()
{
	imshow("",graphcut);
}

void MainWindow::slotSave()
{
	//通过判断当前显示的是哪一个tab来确定保存哪里的结果
	int index = ui.tabWidget->currentIndex();
	//弹出保存对话框
	QString qFileName;
	
	qFileName = QFileDialog::getSaveFileName(this,tr("保存文件"), "", tr("Image Files (*.png *.jpg *.bmp)"));
	if (!qFileName.isNull()) //如果文件名非空，表示正在等待保存
	{
		//保存
		imageProcess.saveResult(index,util.q2s(qFileName));
	} 
	else
	{
		return;
	}
	consoleText = "文件保存完成";
	emit signalShowConsole();
}


void MainWindow::controlGUI(int state)
{
	switch (state)
	{
	case 0: //未打开文件目录
		/*action不可用*/
		ui.action_3->setDisabled(true);
		ui.action_4->setDisabled(true);
		ui.action_2->setDisabled(true);
		/*button不可用*/
		ui.pushButton->setDisabled(true);
		ui.pushButton_2->setDisabled(true);
		ui.pushButton_3->setDisabled(true);
		break;
	case 1:	//未选择影像
		/*action不可用*/
		ui.action_3->setDisabled(true);
		ui.action_4->setDisabled(true);
		ui.action_2->setDisabled(true);
		/*button不可用*/
		ui.pushButton->setDisabled(true);
		ui.pushButton_2->setDisabled(true);
		ui.pushButton_3->setDisabled(true);
		break;
	case 2: //未提取特征
		/*action可用*/
		ui.action_3->setDisabled(false);
		//ui.action_4->setDisabled(false);
		//ui.action_2->setDisabled(false);

		ui.pushButton->setDisabled(false);			//特征提取按钮可用
		break;
	case 3: //未匹配特征
		/*action可用*/
		ui.action_3->setDisabled(false);
		//ui.action_4->setDisabled(false);
		//ui.action_2->setDisabled(false);

		ui.pushButton->setDisabled(false);			//特征提取按钮可用
		ui.pushButton_2->setDisabled(false);		//特征匹配按钮可用
		break;
	case 4: //未对齐影像
		/*action可用*/
		ui.action_3->setDisabled(false);
		ui.action_4->setDisabled(false);
		ui.action_2->setDisabled(false);

		ui.pushButton->setDisabled(false);			//特征提取按钮可用
		ui.pushButton_2->setDisabled(false);		//特征匹配按钮可用
		ui.pushButton_3->setDisabled(false);		//影像对齐按钮可用
		break;
	case 5:
		/*action可用*/
		ui.action_3->setDisabled(false);
		ui.action_4->setDisabled(false);
		ui.action_2->setDisabled(false);

		ui.pushButton->setDisabled(false);			//特征提取按钮可用
		ui.pushButton_2->setDisabled(false);		//特征匹配按钮可用
		ui.pushButton_3->setDisabled(false);		//影像对齐按钮可用
		break;
	}

}
