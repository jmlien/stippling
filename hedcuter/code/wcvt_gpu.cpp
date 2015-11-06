#include "wcvt_gpu.h"

//buil the VOR once
void CVT::vor_GPU(cv::Mat &  img)
{
	//cv::Mat dist(img.size(), CV_32F, cv::Scalar::all(FLT_MAX)); //an image with infinity distance
	//cv::Mat root(img.size(), CV_16U, cv::Scalar::all(USHRT_MAX)); //an image of root index
	//cv::Mat visited(img.size(), CV_8U, cv::Scalar::all(0)); //all unvisited


	////init
	//vector< pair<float, cv::Point> > open;
	//ushort site_id = 0;
	//for (auto& c : cvt.cells)
	//{
	//	cv::Point pix((int)c.site.x, (int)c.site.y);
	//	float d = color2dist(img, pix);
	//	dist.at<float>(pix.x, pix.y) = d;
	//	root.at<ushort>(pix.x, pix.y) = site_id++;
	//	open.push_back( make_pair(d, pix) );
	//	c.coverage.clear();
	//}
	//
	//make_heap(open.begin(), open.end(), compareCell);

	////propagate
	//while (open.empty() == false)
	//{
	//	//
	//	std::pop_heap(open.begin(), open.end(), compareCell);
	//	auto cell = open.back();
	//	auto& cpos = cell.second;
	//	open.pop_back();

	//	//check if the distance from this cell is already updated
	//	if (cell.first > dist.at<float>(cpos.x, cpos.y)) continue;
	//	if (visited.at<uchar>(cpos.x, cpos.y) > 0) continue; //visited
	//	visited.at<uchar>(cpos.x, cpos.y) = 1;

	//	//check the neighbors
	//	for (int dx =-1; dx <= 1; dx++) //x is row
	//	{
	//		int x = cpos.x + dx;
	//		if (x < 0 || x >= img.size().height) continue;
	//		for (int dy = -1; dy <= 1; dy++) //y is column
	//		{
	//			if (dx == 0 && dy == 0) continue; //itself...

	//			int y = cpos.y + dy;
	//			if (y < 0 || y >= img.size().width) continue;
	//			float newd = dist.at<float>(cpos.x, cpos.y) + color2dist(img, cv::Point(x, y));
	//			float oldd = dist.at<float>(x, y);

	//			if (newd < oldd)
	//			{
	//				dist.at<float>(x, y)=newd;
	//				root.at<ushort>(x, y) = root.at<ushort>(cpos.x, cpos.y);
	//				open.push_back(make_pair(newd, cv::Point(x,y)));
	//				std::push_heap(open.begin(), open.end(), compareCell);
	//			}
	//		}//end for dy
	//	}//end for dx
	//}//end while

	////collect cells
	//for (int x = 0; x < img.size().height; x++)
	//{
	//	for (int y = 0; y < img.size().width; y++)
	//	{
	//		ushort rootid = root.at<ushort>(x, y);
	//		cvt.cells[rootid].coverage.push_back(cv::Point(x,y));
	//	}//end y
	//}//end x

	////remove empty cells...
	//int cvt_size = cvt.cells.size();
	//for (int i = 0; i < cvt_size; i++)
	//{
	//	if (cvt.cells[i].coverage.empty())
	//	{
	//		cvt.cells[i] = cvt.cells.back();
	//		cvt.cells.pop_back();
	//		i--;
	//		cvt_size--;
	//	}
	//}//end for i

	//double min;
	//double max;
	//cv::minMaxIdx(dist, &min, &max);
	//cv::Mat adjMap;
	//cv::convertScaleAbs(dist, adjMap, 255 / max);
	////cv::applyColorMap(adjMap, adjMap, cv::COLORMAP_JET);

	//for (auto& c : cvt.cells)
	//{
	//	cv::circle(adjMap, cv::Point(c.site.y, c.site.x), 2, CV_RGB(0, 0, 255), -1);
	//}

	//cv::imshow("dist", adjMap);
	//cv::waitKey(5);
}


void CVT::compute_weighted_cvt_GPU(cv::Mat &  img, std::vector<cv::Point2d> & sites)
{
	//inint 
	int site_size = sites.size();
	cells.resize(site_size);
	for (int i = 0; i < site_size; i++)
	{
		cells[i].site = sites[i];
	}

	float max_dist_moved = FLT_MAX;

	int iteration = 0;
	do
	{
		vor_GPU(img); //compute voronoi
		max_dist_moved = move_sites(img);
		if (debug) std::cout << "[" << iteration << "] max dist moved = " << max_dist_moved << std::endl;
		iteration++;
	} while (max_dist_moved>max_site_displacement && iteration < this->iteration_limit);

	if (debug) cv::waitKey();
}