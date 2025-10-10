#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <vector>

using namespace cv;
using namespace std;

// Simple Sobel edge detection
void sobel(const Mat& input, Mat& output, int num_threads) {
    output = Mat::zeros(input.size(), CV_8UC1);
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for
    for (int y = 1; y < input.rows - 1; y++) {
        for (int x = 1; x < input.cols - 1; x++) {
            int gx = -input.at<uchar>(y-1, x-1) + input.at<uchar>(y-1, x+1) +
                     -2*input.at<uchar>(y, x-1) + 2*input.at<uchar>(y, x+1) +
                     -input.at<uchar>(y+1, x-1) + input.at<uchar>(y+1, x+1);
                     
            int gy = input.at<uchar>(y-1, x-1) + 2*input.at<uchar>(y-1, x) + input.at<uchar>(y-1, x+1) +
                     -input.at<uchar>(y+1, x-1) - 2*input.at<uchar>(y+1, x) - input.at<uchar>(y+1, x+1);
            
            int magnitude = sqrt(gx*gx + gy*gy);
            output.at<uchar>(y, x) = min(255, magnitude);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }
    
    // Load and check image
    Mat color = imread(argv[1]);
    if (color.empty()) {
        cout << "Could not load image!" << endl;
        return -1;
    }
    
    Mat gray;
    cvtColor(color, gray, COLOR_BGR2GRAY);
    
    cout << "=== HPC SCALING ANALYSIS ===" << endl;
    cout << "Image: " << argv[1] << endl;
    cout << "Size: " << gray.cols << " x " << gray.rows << " pixels" << endl;
    cout << "Total pixels: " << gray.cols * gray.rows << endl;
    cout << "Available threads: " << omp_get_max_threads() << endl;
    cout << "=============================" << endl;
    
    Mat result;
    
    // 1. STRONG SCALING (same problem, more threads)
    cout << "\n STRONG SCALING (Fixed problem size, variable threads):" << endl;
    vector<int> thread_counts = {1, 2, 4, 8};
    vector<double> times;
    
    for (int threads : thread_counts) {
        auto start = chrono::high_resolution_clock::now();
        sobel(gray, result, threads);
        auto end = chrono::high_resolution_clock::now();
        
        double time = chrono::duration<double>(end - start).count();
        times.push_back(time);
        
        cout << threads << " threads: " << time << "s";
        if (threads > 1) {
            double speedup = times[0] / time;
            double efficiency = speedup / threads * 100;
            cout << " (Speedup: " << speedup << "x, Efficiency: " << efficiency << "%)";
        }
        cout << endl;
    }
    
    // 2. WEAK SCALING (ordered by pixels: smallest to largest)
    cout << "\n WEAK SCALING (Ordered by problem size, threads increase accordingly):" << endl;
    vector<string> images = {"try.jpg", "tryit.jpg", "1024_2.png", "trail.jpg"};  // Ordered by pixel count
    vector<int> thread_counts_weak = {1, 2, 4, 8};
    
    // Show expected ordering
    cout << "Order: try.jpg(774K) → tryit.jpg(810K) → 1024_2.png(1M) → trail.jpg(20M)" << endl;
    
    for (int i = 0; i < images.size(); i++) {
        string img_file = images[i];
        int threads = thread_counts_weak[i];
        
        // Load different image
        Mat color_weak = imread(img_file);
        if (color_weak.empty()) {
            cout << "Could not load " << img_file << ", skipping..." << endl;
            continue;
        }
        
        Mat gray_weak;
        cvtColor(color_weak, gray_weak, COLOR_BGR2GRAY);
        
        auto start = chrono::high_resolution_clock::now();
        sobel(gray_weak, result, threads);
        auto end = chrono::high_resolution_clock::now();
        
        double time = chrono::duration<double>(end - start).count();
        int pixels = gray_weak.cols * gray_weak.rows;
        double throughput = pixels / time / 1000000;  // Mpixels/s
        double work_per_thread = (double)pixels / threads / 1000000;  // Mpixels per thread
        
        cout << threads << " threads, " << img_file << " (" << gray_weak.cols << "x" << gray_weak.rows << ", " << pixels/1000 << "K pixels): ";
        cout << time << "s, " << throughput << " Mpixels/s, " << work_per_thread << " Mpixels/thread" << endl;
    }
    
    // Summary
    cout << "\n ANALYSIS SUMMARY:" << endl;
    cout << "✓ Strong scaling: Same work (" << gray.cols*gray.rows << " pixels), more threads" << endl;
    cout << "✓ Weak scaling: Work scales with thread count" << endl;
    cout << "✓ Best speedup: " << times[0]/times.back() << "x with " << thread_counts.back() << " threads" << endl;
    
    return 0;
}
