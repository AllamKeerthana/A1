#include <opencv2/opencv.hpp>  // OpenCV for image read/write
#include <iostream>            // For cout, cerr
#include <vector>              // std::vector for dynamic arrays
#include <thread>              // std::thread for manual parallelism
#include <cmath>               // sqrt for Sobel filter
#include <chrono>              // High resolution timer

using namespace std;
using uchar = unsigned char;   // Alias for pixel intensity type

// --- Single-threaded Sobel filter ---
void sobelFilterSingleThread(const cv::Mat& gray, cv::Mat& edges) {
    // Process all inner pixels (excluding border rows/columns)
    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x - 1) + gray.at<uchar>(y - 1, x + 1)
            - 2 * gray.at<uchar>(y, x - 1) + 2 * gray.at<uchar>(y, x + 1)
            - gray.at<uchar>(y + 1, x - 1) + gray.at<uchar>(y + 1, x + 1);

            int gy = gray.at<uchar>(y - 1, x - 1) + 2 * gray.at<uchar>(y - 1, x) + gray.at<uchar>(y - 1, x + 1)
            - gray.at<uchar>(y + 1, x - 1) - 2 * gray.at<uchar>(y + 1, x) - gray.at<uchar>(y + 1, x + 1);

            int mag = static_cast<int>(sqrt(gx * gx + gy * gy)); // convert gradient magnitude to int
            edges.at<uchar>(y, x) = (mag > 255) ? 255 : mag;     // clamp to 255
        }
    }

    // --- Set border pixels to 0 after computing inner pixels ---
    for (int x = 0; x < gray.cols; ++x) {
        edges.at<uchar>(0, x) = 0;                     // top row
        edges.at<uchar>(gray.rows - 1, x) = 0;        // bottom row
    }
    for (int y = 0; y < gray.rows; ++y) {
        edges.at<uchar>(y, 0) = 0;                     // left column
        edges.at<uchar>(y, gray.cols - 1) = 0;        // right column
    }
}

// --- Parallel Sobel function executed by each thread ---
void sobelFilterThread(const cv::Mat& gray, cv::Mat& edges, int startRow, int endRow, int thread_id) {
    // Compute only inner pixels in assigned rows (skip top/bottom borders)
    for (int y = max(1, startRow); y < min(endRow, gray.rows - 1); ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x - 1) + gray.at<uchar>(y - 1, x + 1)
            - 2 * gray.at<uchar>(y, x - 1) + 2 * gray.at<uchar>(y, x + 1)
            - gray.at<uchar>(y + 1, x - 1) + gray.at<uchar>(y + 1, x + 1);

            int gy = gray.at<uchar>(y - 1, x - 1) + 2 * gray.at<uchar>(y - 1, x) + gray.at<uchar>(y - 1, x + 1)
            - gray.at<uchar>(y + 1, x - 1) - 2 * gray.at<uchar>(y + 1, x) - gray.at<uchar>(y + 1, x + 1);

            int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
            edges.at<uchar>(y, x) = (mag > 255) ? 255 : mag;
        }
    }
    cout << "Thread " << thread_id << " processed rows " << startRow << " to " << (endRow - 1) << endl;
}

int main() {
    // Hardcoded paths and number of threads
    string input_path = "/home/kKethana/Pictures/verbal.jpeg";
    string output_path_serial = "/home/kKethana/Documents/HPC/project/ME_serial.png";
    string output_path_parallel = "/home/kKethana/Documents/HPC/project/ME_parallel.png";
    int num_threads = 4;

    // Load the image as grayscale
    cv::Mat gray = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        cerr << "Error: Could not open image at " << input_path << endl;
        return -1;
    }
    cout << "Loaded image: " << input_path << " Size: " << gray.cols << "x" << gray.rows << endl;

    // --- Single-threaded Sobel ---
    cv::Mat edges_single = cv::Mat::zeros(gray.size(), gray.type());
    auto start_serial = chrono::high_resolution_clock::now();
    sobelFilterSingleThread(gray, edges_single);
    auto end_serial = chrono::high_resolution_clock::now();
    double serial_time = chrono::duration<double>(end_serial - start_serial).count();
    cout << "Single-threaded Sobel took: " << serial_time << " seconds." << endl;
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
    for (auto& t : threads) t.join();
    auto end_parallel = chrono::high_resolution_clock::now();
    double parallel_time = chrono::duration<double>(end_parallel - start_parallel).count();
    cout << "Parallel Sobel took: " << parallel_time << " seconds." << endl;
    // Save parallel result
    if (!cv::imwrite(output_path_parallel, edges_parallel)) {
        cerr << "Failed to save parallel output image." << endl;
        return -1;
    }
    cout << "Parallel edge image saved to: " << output_path_parallel << endl;

    // --- Set borders to 0 after parallel processing ---
    for (int x = 0; x < gray.cols; ++x) {
        edges_parallel.at<uchar>(0, x) = 0;
        edges_parallel.at<uchar>(gray.rows - 1, x) = 0;
    }
    for (int y = 0; y < gray.rows; ++y) {
        edges_parallel.at<uchar>(y, 0) = 0;
        edges_parallel.at<uchar>(y, gray.cols - 1) = 0;
    }

    cv::imwrite(output_path_parallel, edges_parallel);

    // --- Compare outputs ---
    bool images_equal = cv::countNonZero(edges_single != edges_parallel) == 0;
    cout << "Outputs match: " << (images_equal ? "Yes" : "No") << endl;

    // --- Speedup ---
    if (parallel_time > 0.0)
        cout << "Speedup (serial/parallel): " << serial_time / parallel_time << endl;

    return 0;
}
