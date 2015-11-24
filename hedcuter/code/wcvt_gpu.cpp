#include "wcvt_gpu.h"

std::vector<VorCell> CVT::cells;

GLuint cones;
float rotateY;
GLuint fbo, depthBuffer, texture;
cv::Mat input_image;


void idle_GPU(void)
{
	glutPostRedisplay();
}

void keyboard_GPU(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 'r':
			rotateY += 10.0;
			if (rotateY > 360 || rotateY < -360) rotateY = 0.0;
		break;

		default:
		break;
	}
}



//buil the VOR once
void CVT::vor_GPU(cv::Mat &  img)
{
	cv::Mat dist(img.size(), CV_32F, cv::Scalar::all(FLT_MAX)); //an image with infinity distance
	cv::Mat root(img.size(), CV_16U, cv::Scalar::all(USHRT_MAX)); //an image of root index
	cv::Mat visited(img.size(), CV_8U, cv::Scalar::all(0)); //all unvisited

	//init
	std::vector< std::pair<float, cv::Point> > open;
	ushort site_id = 0;
	for (auto& c : this->cells)
	{
		cv::Point pix((int)c.site.x, (int)c.site.y);
		float d = color2dist(img, pix);
		dist.at<float>(pix.x, pix.y) = d;
		root.at<ushort>(pix.x, pix.y) = site_id++;
		open.push_back( std::make_pair(d, pix) );
		c.coverage.clear();
	}
	
	make_heap(open.begin(), open.end(), compareCell);

	//propagate
	while (open.empty() == false)

	{
		std::pop_heap(open.begin(), open.end(), compareCell);
		auto cell = open.back();
		auto& cpos = cell.second;
		open.pop_back();

		//check if the distance from this cell is already updated
		if (cell.first > dist.at<float>(cpos.x, cpos.y)) continue;
		if (visited.at<uchar>(cpos.x, cpos.y) > 0) continue; //visited
		visited.at<uchar>(cpos.x, cpos.y) = 1;

		//check the neighbors
		for (int dx =-1; dx <= 1; dx++) //x is row
		{
			int x = cpos.x + dx;
			if (x < 0 || x >= img.size().height) continue;
			for (int dy = -1; dy <= 1; dy++) //y is column
			{
				if (dx == 0 && dy == 0) continue; //itself...

				int y = cpos.y + dy;
				if (y < 0 || y >= img.size().width) continue;
				float newd = dist.at<float>(cpos.x, cpos.y) + color2dist(img, cv::Point(x, y));
				float oldd = dist.at<float>(x, y);

				if (newd < oldd)
				{
					dist.at<float>(x, y)=newd;
					root.at<ushort>(x, y) = root.at<ushort>(cpos.x, cpos.y);
					open.push_back(std::make_pair(newd, cv::Point(x,y)));
					std::push_heap(open.begin(), open.end(), compareCell);
				}
			}//end for dy
		}//end for dx
	}//end while

	//collect cells
	for (int x = 0; x < img.size().height; x++)
	{
		for (int y = 0; y < img.size().width; y++)
		{
			ushort rootid = root.at<ushort>(x, y);
			this->cells[rootid].coverage.push_back(cv::Point(x,y));
		}//end y
	}//end x

	//remove empty cells...
	int cvt_size = this->cells.size();
	for (int i = 0; i < cvt_size; i++)
	{
		if (this->cells[i].coverage.empty())
		{
			this->cells[i] = this->cells.back();
			this->cells.pop_back();
			i--;
			cvt_size--;
		}
	}//end for i

	double min;
	double max;
	cv::minMaxIdx(dist, &min, &max);
	cv::Mat adjMap;
	cv::convertScaleAbs(dist, adjMap, 255 / max);
	//cv::applyColorMap(adjMap, adjMap, cv::COLORMAP_JET);

	for (auto& c : this->cells)
	{
		cv::circle(adjMap, cv::Point(c.site.y, c.site.x), 2, CV_RGB(0, 0, 255), -1);
	}

	cv::imshow("dist", adjMap);
	cv::waitKey(5);
}


void CVT::compute_weighted_cvt_GPU(cv::Mat &  img, std::vector<cv::Point2d> & sites)
{
	//init 
	int site_size = sites.size();
	cells.resize(site_size);
	for (int i = 0; i < site_size; i++)
	{
		cells[i].site = sites[i];
	}

	float max_dist_moved = FLT_MAX;

	run_GPU(argc_GPU, argv_GPU, img);

	glDeleteFramebuffersEXT(1, &fbo);
	glDeleteRenderbuffersEXT(1, &depthBuffer);
	/*
	int iteration = 0;
	do
	{
		vor_GPU(img); //compute voronoi	
		max_dist_moved = move_sites(img);
		if (debug) std::cout << "[" << iteration << "] max dist moved = " << max_dist_moved << std::endl;
		iteration++;
	} while (max_dist_moved>max_site_displacement && iteration < this->iteration_limit);

	if (debug) cv::waitKey();
	*/
}

void CVT::run_GPU(int argc, char**argv, cv::Mat img)
{
	//Init opengl
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(img.size().width, img.size().height);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("Image");

	init_GPU(img);

	glutDisplayFunc(display_GPU);
	glutKeyboardFunc(keyboard_GPU);
	glutIdleFunc(idle_GPU);

	glutMainLoop();
}

void CVT::init_GPU(cv::Mat img)
{

	
	glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_ALWAYS);
	//glDepthMask(GL_TRUE);

	rotateY = 0.0;
	input_image = img;
	
	cones = createDisplayList_GPU();

	glewInit();

	//Set up the frame buffer
	glGenFramebuffersEXT(1, &fbo);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);

	//Generate depth buffer using render buffer object
	glGenRenderbuffersEXT(1, &depthBuffer);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, img.size().width, img.size().height);

	//Attach
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER_EXT, depthBuffer);

	//Generate Texture 
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.size().width, img.size().height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	//Attach 2D texture to frame buffer
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, texture, 0);

	GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	if (status != GL_FRAMEBUFFER_COMPLETE_EXT) { std::cout << "Frame buffer error :" << status << std::endl; exit(-1); }
}
void CVT::display_GPU(void)
{
	//Drawing in the frame buffer
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, input_image.size().width, input_image.size().height);
	
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glTranslatef(0.0, 0.0, -0.8);
	glRotatef(rotateY, 0.0, 1.0, 0.0);
	
	//Draw discrete voronoi diagram
	glCallList(cones);		

	//Move the sites

	glPopAttrib();
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	
	//Rendering part
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glTranslatef(0.0, 0.0, -0.8);
	glRotatef(rotateY, 0.0, 1.0, 0.0);
	
	glBindTexture(GL_TEXTURE_2D, texture);
	glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0, -1.0, 0.0);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0, 1.0, 0.0);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0, 1.0, 0.0);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0, -1.0, 0.0);
	glEnd();
	glDisable(GL_TEXTURE_2D);

	glutSwapBuffers();
}


GLuint CVT::createDisplayList_GPU()	//*****
{
	//Cone information
	GLuint coneList;
	GLdouble base = (float)input_image.size().height/2.0;		//*****
	GLdouble height = 0.1;	//*****
	GLint slices = 50;		//*****
	GLint stacks = 50;

	//Image information
	cv::Mat grayscale(input_image.size().width, input_image.size().height, CV_LOAD_IMAGE_GRAYSCALE);
	float d = 0.0;
	unsigned int r = 0, g = 0, b = 0;

	coneList = glGenLists(1);
	glNewList(coneList, GL_COMPILE);

	for (int i = 0; i < cells.size(); i++)
	{
		cv::Point pix(cells[i].site.x, cells[i].site.y);
		d = color2dist(grayscale, pix);
		
		r = input_image.at<cv::Vec3b>(pix.x, pix.y)[2];
		g = input_image.at<cv::Vec3b>(pix.x, pix.y)[1];
		b = input_image.at<cv::Vec3b>(pix.x, pix.y)[0];

		glPushMatrix();	
		//Convert opengl coordinates to opencv coordinates
		glScalef(2.0 / input_image.size().width, 2.0 / input_image.size().height, 1.0);
		glTranslatef(-(float)input_image.size().width / 2.0, (float)input_image.size().height / 2.0, 0.0);
		glRotatef(180.0, 1.0, 0.0, 0.0);

		glColor3ub(r, g, b);
		glTranslatef(cells[i].site.y*1.0f, cells[i].site.x*1.0f, 0.0);
		glutSolidCone(base, height, slices, stacks);
		glPopMatrix();
	}

	glEndList();

	return coneList;
}
