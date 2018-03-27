/**
 * @file CannyDetector_Demo.cpp
 * @brief Sample code showing how to detect edges using the Canny Detector
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <vector>

using namespace cv;

#define FILE_NAME_LENGTH 100

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
const char* window_name = "Edge Map";
char file[FILE_NAME_LENGTH] = {'\0'};
bool QUIET = true;

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
static void CannyThreshold(int, void*)
{
    src.copyTo( dst, detected_edges);
    
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );

    /// Canny detector
		if(QUIET){
			lowThreshold = 50;
		}
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    src.copyTo( dst, detected_edges);
		
		std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
		char temp[100]={'\0'};
		int i,size=sizeof("canny_")/sizeof(char);
		strncpy(temp,"canny_",size);
	
		strcat(temp,file);
		imwrite(temp, dst, compression_params);

    if(!QUIET){
			imshow( window_name, dst );
		}
}

/**
 * @function main
 */
int main( int argc, char** argv )
{
	if(!*argv[1]){
		fprintf(stderr,"danger\n");
		return 1;
	}
	if(argc==3){
		printf("(%d:%c)\n",argc,*argv[argc-1]);
		if(*argv[argc-1]=='0'){
			QUIET=false;
		}
	}

	int i;
	for(i=0;i<FILE_NAME_LENGTH;++i){
		file[i]=*(argv[1]+i);
	}
	file[i+1]='\0';
  src = imread( file, IMREAD_COLOR ); // Load an image

  if( src.empty() )
    { return -1; }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  cvtColor( src, src_gray, COLOR_BGR2GRAY );

  if(!QUIET){
		namedWindow( window_name, WINDOW_AUTOSIZE );
		createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
	}

  CannyThreshold(0, 0);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
}
