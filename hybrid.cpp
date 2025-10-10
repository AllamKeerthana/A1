#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <cstring>

/* 
MINIMAL CHANGES MADE TO AVOID GHOST ROW DEADLOCK:
1. Modified sobelOnChunkOpenMP() to skip boundary rows that need ghost data
2. Disabled exchangeGhostRows() to avoid MPI_Sendrecv deadlock
3. Added boundary-safe processing logic
*/

using namespace std;
using uchar = unsigned char;

// Sobel filter parallelized with OpenMP inside each MPI rank on local image chunk
void sobelOnChunkOpenMP(const cv::Mat& gray, cv::Mat& edges, int start_row, int end_row, int rank, int size) {
    int cols = gray.cols;
    
    // Skip boundaries that need ghost rows - SIMPLE APPROACH
    int safe_start = start_row;
    int safe_end = end_row;
    
    // Skip first row if we're the first process
    if (rank == 0) safe_start = max(1, start_row);
    
    // Skip last row if we're the last process  
    if (rank == size - 1) safe_end = min(end_row, gray.rows - 1);

    #pragma omp parallel for schedule(dynamic)
    for (int y = safe_start; y < safe_end; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int gx = -gray.at<uchar>(y - 1, x - 1) + gray.at<uchar>(y - 1, x + 1)
                     - 2 * gray.at<uchar>(y, x - 1) + 2 * gray.at<uchar>(y, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) + gray.at<uchar>(y + 1, x + 1);

            int gy = gray.at<uchar>(y - 1, x - 1) + 2 * gray.at<uchar>(y - 1, x) + gray.at<uchar>(y - 1, x + 1)
                     - gray.at<uchar>(y + 1, x - 1) - 2 * gray.at<uchar>(y + 1, x) - gray.at<uchar>(y + 1, x + 1);

            int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
            edges.at<uchar>(y, x) = (mag > 255) ? 255 : mag;
        }
        edges.at<uchar>(y, 0) = 0;
        edges.at<uchar>(y, cols - 1) = 0;
    }
    
    // Set skipped boundary rows to black
    if (rank == 0 && start_row == 0) {
        for (int x = 0; x < cols; x++) {
            edges.at<uchar>(0, x) = 0;
        }
    }
}

// SIMPLIFIED: Skip ghost row exchange to avoid deadlock
void exchangeGhostRows(MPI_Comm comm, cv::Mat& local_gray, int rank, int size) {
    // DISABLED: Ghost row exchange causes deadlock
    // Instead we skip boundary pixels that need ghost data
    // This is simpler and avoids all communication issues
    
    cout << "Rank " << rank << ": Skipping ghost row exchange (boundary-safe approach)" << endl;
    
   
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    string img_path = argv[1];
    cv::Mat gray;
    int rows = 0, cols = 0;

    if (rank == 0) {
        cv::Mat color_img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (color_img.empty()) {
            cerr << "Could not load image: " << img_path << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cv::cvtColor(color_img, gray, cv::COLOR_BGR2GRAY);
        rows = gray.rows;
        cols = gray.cols;
        cout << "Image loaded: " << cols << "x" << rows << " pixels" << endl;
    }

    // Broadcast image dimensions to all ranks
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate chunk sizes and offsets per rank
    vector<int> sendcounts(size), displs(size);
    int base_rows = rows / size;
    int remainder = rows % size;
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = base_rows + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_rows = sendcounts[rank];
    int local_with_ghosts = local_rows + 2;
    if (rank == 0) local_with_ghosts = local_rows + 1;
    if (rank == size - 1) local_with_ghosts = local_rows + 1;

    vector<uchar> local_data(local_with_ghosts * cols);

    // Scatter the image chunks with ghost rows
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            int chunk_start = displs[i];
            int chunk_size = sendcounts[i];
            int rows_to_copy = chunk_size + 2;
            if (i == 0 || i == size - 1) rows_to_copy = chunk_size + 1;
            int start_send = (i == 0) ? chunk_start : chunk_start - 1;

            vector<uchar> buffer(rows_to_copy * cols);
            for (int r = 0; r < rows_to_copy; r++) {
                if (start_send + r >= 0 && start_send + r < rows) {
                    memcpy(&buffer[r * cols], gray.ptr<uchar>(start_send + r), cols * sizeof(uchar));
                }
            }

            if (i == 0) {
                memcpy(local_data.data(), buffer.data(), rows_to_copy * cols);
            } else {
                MPI_Send(buffer.data(), rows_to_copy * cols, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(local_data.data(), local_with_ghosts * cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cv::Mat local_gray(local_with_ghosts, cols, CV_8UC1, local_data.data());
    cv::Mat local_edges = cv::Mat::zeros(local_gray.size(), CV_8UC1);

    // Synchronize all ranks before start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Exchange halo rows - DISABLED to avoid deadlock
    exchangeGhostRows(MPI_COMM_WORLD, local_gray, rank, size);

    // Parallel Sobel filter using OpenMP on local chunk - skip boundaries  
    int actual_start = (rank == 0) ? 0 : 1;
    int actual_end = (rank == size - 1) ? local_with_ghosts : local_with_ghosts - 1;
    
    sobelOnChunkOpenMP(local_gray, local_edges, actual_start, actual_end, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    // Prepare data for gathering (remove ghost rows)
    vector<uchar> local_result(local_rows * cols);
    int copy_start = (rank == 0) ? 0 : 1;
    for (int r = 0; r < local_rows; ++r) {
        memcpy(&local_result[r * cols], local_edges.ptr<uchar>(r + copy_start), cols * sizeof(uchar));
    }

    vector<int> recvcounts(size), recvdispls(size);
    offset = 0;
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = sendcounts[i] * cols;
        recvdispls[i] = offset;
        offset += recvcounts[i];
    }
    vector<uchar> full_result;
    if (rank == 0) full_result.resize(rows * cols);

    // Gather all results at root
    MPI_Gatherv(local_result.data(), local_rows * cols, MPI_UNSIGNED_CHAR,
                full_result.data(), recvcounts.data(), recvdispls.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        cv::Mat edges(rows, cols, CV_8UC1, full_result.data());

        // Zero final output image boundaries
        for (int x = 0; x < cols; ++x) {
            edges.at<uchar>(0, x) = 0;
            edges.at<uchar>(rows - 1, x) = 0;
        }
        for (int y = 0; y < rows; ++y) {
            edges.at<uchar>(y, 0) = 0;
            edges.at<uchar>(y, cols - 1) = 0;
        }

        cv::imwrite("edges_hybrid.png", edges);
        cout << "Hybrid MPI+OpenMP Sobel completed. Time: " << (end - start) << " s" << endl;
    }

    MPI_Finalize();
    return 0;
}
