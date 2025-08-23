#include <opencv2/opencv.hpp>  // OpenCV for image read/write
#include <iostream>            // For cout, cerr
#include <vector>              // std::vector for dynamic arrays
#include <thread>              // std::thread for manual parallelism
#include <cmath>               // sqrt for Sobel filter
#include <chrono>              // High resolution timer

using namespace std;
using uchar = unsigned char;   // Alias for pixel intensity type

// Single-threaded Sobel filter applied to the whole image
void sobelFilterSingleThread(const cv::Mat& gray, cv::Mat& edges) {
    for (int y = 0; y < gray.rows; ++y) {
        if (y == 0 || y == gray.rows - 1) {
            for (int x = 0; x < gray.cols; ++x)
                edges.at<uchar>(y, x) = 0;
            continue;
        }
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x - 1) + gray.at<uchar>(y - 1, x + 1)
                     - 2 * gray.at<uchar>(y, x - 1) + 2 * gray.at<uchar>(y, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) + gray.at<uchar>(y + 1, x + 1);
            int gy = gray.at<uchar>(y - 1, x - 1) + 2 * gray.at<uchar>(y - 1, x) + gray.at<uchar>(y - 1, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) - 2 * gray.at<uchar>(y + 1, x) - gray.at<uchar>(y + 1, x + 1);
            int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
            edges.at<uchar>(y, x) = (mag > 255) ? 255 : mag;
        }
        edges.at<uchar>(y, 0) = 0;
        edges.at<uchar>(y, gray.cols - 1) = 0;
    }
}

// Function executed by each thread: apply Sobel filter to assigned rows
void sobelFilterThread(const cv::Mat& gray, cv::Mat& edges, int startRow, int endRow, int thread_id) {
    for (int y = startRow; y < endRow; ++y) {
        if (y == 0 || y == gray.rows - 1) {
            for (int x = 0; x < gray.cols; ++x)
                edges.at<uchar>(y, x) = 0;
            continue;
        }
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x - 1) + gray.at<uchar>(y - 1, x + 1)
                     - 2 * gray.at<uchar>(y, x - 1) + 2 * gray.at<uchar>(y, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) + gray.at<uchar>(y + 1, x + 1);
            int gy = gray.at<uchar>(y - 1, x - 1) + 2 * gray.at<uchar>(y - 1, x) + gray.at<uchar>(y - 1, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) - 2 * gray.at<uchar>(y + 1, x) - gray.at<uchar>(y + 1, x + 1);
            int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
            edges.at<uchar>(y, x) = (mag > 255) ? 255 : mag;
        }
        edges.at<uchar>(y, 0) = 0;
        edges.at<uchar>(y, gray.cols - 1) = 0;
    }
    cout << "Thread " << thread_id << " processed rows " << startRow << " to " << (endRow - 1) << endl;
}

int main() {
    // Hardcoded paths and number of threads
    string input_path = "/home/kKethana/Pictures/verbal.jpeg";
    string output_path_serial = "/home/kKethana/Documents/HPC/project/ME_serial.png";
    string output_path_parallel = "/home/kKethana/Documents/HPC/project/ME_parallel.png";
    int num_threads = 4;  // You can change this to desired thread count

    // Load image as grayscale using OpenCV
    cv::Mat gray = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        cerr << "Error: Could not open or find the image at " << input_path << endl;
        return -1;
    }
    cout << "Loaded image: " << input_path << " Size: " << gray.cols << "x" << gray.rows << endl;

    // --- Single-threaded Sobel ---
    cv::Mat edges_single = cv::Mat::zeros(gray.size(), gray.type());
    auto start_serial = chrono::high_resolution_clock::now();
    sobelFilterSingleThread(gray, edges_single);
    auto end_serial = chrono::high_resolution_clock::now();
    double serial_time = chrono::duration<double>(end_serial - start_serial).count();
    cout << "Single-threaded Sobel edge detection took: " << serial_time << " seconds." << endl;

    // Save single-threaded result
    if (!cv::imwrite(output_path_serial, edges_single)) {
        cerr << "Failed to save single-threaded output image." << endl;
        return -1;
    }
    cout << "Single-threaded edge image saved to: " << output_path_serial << endl;

    // --- Parallel Sobel ---
    cv::Mat edges_parallel = cv::Mat::zeros(gray.size(), gray.type());
    vector<thread> threads;
    int rows_per_thread = gray.rows / num_threads;
    int remainder = gray.rows % num_threads;
    int start_row = 0;

    auto start_parallel = chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        int end_row = start_row + rows_per_thread + (t < remainder ? 1 : 0);
        threads.emplace_back(sobelFilterThread, cref(gray), ref(edges_parallel), start_row, end_row, t);
        start_row = end_row;
    }
    for (auto& t : threads) {
        t.join();
    }

    auto end_parallel = chrono::high_resolution_clock::now();
    double parallel_time = chrono::duration<double>(end_parallel - start_parallel).count();
    cout << "Parallel Sobel edge detection took: " << parallel_time << " seconds." << endl;

    // Save parallel result
    if (!cv::imwrite(output_path_parallel, edges_parallel)) {
        cerr << "Failed to save parallel output image." << endl;
        return -1;
    }
    cout << "Parallel edge image saved to: " << output_path_parallel << endl;

    // --- Compare outputs ---
    bool images_equal = cv::countNonZero(edges_single != edges_parallel) == 0;
    cout << "Outputs match: " << (images_equal ? "Yes" : "No") << endl;

    // --- Print speedup ---
    if (parallel_time > 0.0)
        cout << "Speedup (serial/parallel): " << serial_time / parallel_time << endl;

    return 0;
}
