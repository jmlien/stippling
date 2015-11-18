#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <limits>
#include <iomanip>
#include <numeric>
#include <iostream>

#undef check

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/contrib/contrib.hpp"


struct VorCell
{
	VorCell(){}

	VorCell(const VorCell& other)
	{
		site = other.site;
		coverage = other.coverage;
	}

	cv::Point site;
	std::list<cv::Point> coverage;
};

struct CVT
{
public:

	CVT()
	{
		iteration_limit = 100;
		max_site_displacement = 1.01f;
		average_termination = false;
//		subpixel_level = false;
		debug = false;
	}

	void compute_weighted_cvt(cv::Mat &  img, std::vector<cv::Point2d> & pts);
	void compute_weighted_cvt_GPU(cv::Mat &  img, std::vector<cv::Point2d> & sites);

	const std::vector<VorCell> & getCells() const 
	{
		return this->cells;
	}

	int iteration_limit;       //max number of iterations when building cvf
	float max_site_displacement; //max tolerable site displacement in each iteration. 
	bool average_termination;
	///int subpixel_level;

	bool debug;

private:
	
	std::vector<VorCell> cells;

	void vor(cv::Mat &  img);
	void vor_GPU(cv::Mat &  img);

	//convert a color intensity to distance between 0~1
	inline float color2dist(cv::Mat &  img, cv::Point& p)
	{
		//note: 256 is used here instead of 255 to prevent 0 distance.
		return (256 - img.at<uchar>(p.x, p.y))*1.0f / 256;
	}

	//move the site to the center of its coverage
	inline float move_sites(cv::Mat &  img, VorCell & cell)
	{
		if (cell.coverage.empty()) std::cout << "! Error: cell.coverage " << cell.site << " size = " << cell.coverage.size() << std::endl;

		//compute weighted average
		float total = 0;
		cv::Point2d new_pos(0, 0);
		for (auto& c : cell.coverage)
		{
			float d = color2dist(img, c);
			new_pos.x += d*c.x;
			new_pos.y += d*c.y;
			total += d;
		}

		//normalize
		new_pos.x /= total;
		new_pos.y /= total;

		//update
		float dist = fabs(new_pos.x - cell.site.x) + fabs(new_pos.y - cell.site.y); //manhattan dist
		cell.site = new_pos;

		//done
		return dist;
	}

	//move the sites to the centers of their coverages
	inline float move_sites(cv::Mat &  img)
	{
		float max_offset = 0;
		if (average_termination)
		{
			for (auto& cell : this->cells)
			{
				//cout << "coverage size=" << cvt.cells[607].coverage.size() << endl;
				float offset = move_sites(img, cell);
				max_offset += offset;
			}

			max_offset /= this->cells.size();
		}
		else
		{
			for (auto& cell : this->cells)
			{
				//cout << "coverage size=" << cvt.cells[607].coverage.size() << endl;
				float offset = move_sites(img, cell);
				if (offset > max_offset)
					max_offset = offset;
			}
		}
		return max_offset;
	}
};