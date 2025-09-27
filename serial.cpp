#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using uchar = unsigned char;

// Serial Sobel function
void sobelFilterSingleThread(const cv::Mat& gray, cv::Mat& edges) {
    int rows = gray.rows, cols = gray.cols;
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x - 1) + gray.at<uchar>(y - 1, x + 1)
                     - 2 * gray.at<uchar>(y, x - 1) + 2 * gray.at<uchar>(y, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) + gray.at<uchar>(y + 1, x + 1);

            int gy = gray.at<uchar>(y - 1, x - 1) + 2 * gray.at<uchar>(y - 1, x) + gray.at<uchar>(y - 1, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) - 2 * gray.at<uchar>(y + 1, x) - gray.at<uchar>(y + 1, x + 1);

            int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
            edges.at<uchar>(y, x) = (mag > 255) ? 255 : mag;
        }
    }

    // zero borders
    for (int x = 0; x < cols; ++x) {
        edges.at<uchar>(0, x) = 0;
        edges.at<uchar>(rows - 1, x) = 0;
    }
    for (int y = 0; y < rows; ++y) {
        edges.at<uchar>(y, 0) = 0;
        edges.at<uchar>(y, cols - 1) = 0;
    }
}

// OpenMP Sobel function
void sobelFilterOpenMP(const cv::Mat& gray, cv::Mat& edges) {
    int rows = gray.rows, cols = gray.cols;

    #pragma omp parallel for schedule(dynamic)
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x - 1) + gray.at<uchar>(y - 1, x + 1)
                     - 2 * gray.at<uchar>(y, x - 1) + 2 * gray.at<uchar>(y, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) + gray.at<uchar>(y + 1, x + 1);

            int gy = gray.at<uchar>(y - 1, x - 1) + 2 * gray.at<uchar>(y - 1, x) + gray.at<uchar>(y - 1, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) - 2 * gray.at<uchar>(y + 1, x) - gray.at<uchar>(y + 1, x + 1);

            int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
            edges.at<uchar>(y, x) = (mag > 255) ? 255 : mag;
        }
    }

    // zero borders as well
    #pragma omp parallel for
    for (int x = 0; x < cols; ++x) {
        edges.at<uchar>(0, x) = 0;
        edges.at<uchar>(rows - 1, x) = 0;
    }
    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        edges.at<uchar>(y, 0) = 0;
        edges.at<uchar>(y, cols - 1) = 0;
    }
}

// Other necessary MPI and hybrid functions (halo exchange, local chunk Sobel with OpenMP) 
// can be implemented as detailed in our prior exchanges.

// Helper to convert color image to grayscale, run all tests and measure times
int main() {
    string img_path = "1024_2.png";

    cv::Mat color_img = cv::imread(img_path, cv::IMREAD_COLOR);
    if(color_img.empty()){
        cerr << "Error loading image: " << img_path << endl;
        return -1;
    }
    
    cv::Mat gray_img;
    cv::cvtColor(color_img, gray_img, cv::COLOR_BGR2GRAY);

    int rows = gray_img.rows;
    int cols = gray_img.cols;
    
    cout << "Image loaded: " << cols << "x" << rows << " pixels" << endl;

    // Prepare containers for serial and OpenMP
    cv::Mat edges_serial = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat edges_openmp = cv::Mat::zeros(rows, cols, CV_8UC1);

    // Serial run
    auto start = chrono::high_resolution_clock::now();
    sobelFilterSingleThread(gray_img, edges_serial);
    auto end = chrono::high_resolution_clock::now();
    double serial_time = chrono::duration<double>(end - start).count();
    cout << "Serial Sobel time: " << serial_time << " s" << endl;
    cv::imwrite("edges_serial.png", edges_serial);

    // OpenMP run
    start = chrono::high_resolution_clock::now();
    sobelFilterOpenMP(gray_img, edges_openmp);
    end = chrono::high_resolution_clock::now();
    double openmp_time = chrono::duration<double>(end - start).count();
    cout << "OpenMP Sobel time: " << openmp_time << " s" << endl;
    cv::imwrite("edges_openmp.png", edges_openmp);

    cout << "Edge detection completed. Output images saved as edges_serial.png and edges_openmp.png" << endl;

    return 0;
}
