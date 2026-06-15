#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <cmath>

/**
 * @brief Compute PSNR (Peak Signal-to-Noise Ratio) between two BGR8 images
 * 
 * Compares a reference image against a test image, computing the mean squared error
 * and converting to PSNR in decibels. Handles image resizing if dimensions differ.
 * 
 * @param reference_bgr8 Reference image (CV_8UC3 or CV_8UC4)
 * @param test_bgr8 Test image to compare (CV_8UC3 or CV_8UC4)
 * @return PSNR value in dB, NaN if images are empty, infinity if images are identical
 * 
 * @note
 * - Automatically resizes reference image to match test image size if needed
 * - Uses appropriate interpolation (INTER_AREA for downsampling, INTER_LINEAR for upsampling)
 * - Computes MSE across all 3 channels (RGB)
 * - Peak signal value assumed to be 255 (8-bit unsigned integer)
 * 
 * Formula: PSNR = 10 * log10((255^2) / MSE)
 */
inline double computeImagePsnrDb(const cv::Mat &reference_bgr8, const cv::Mat &test_bgr8)
{
    if (reference_bgr8.empty() || test_bgr8.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Resize reference image to match test image dimensions if needed
    cv::Mat reference_resized;
    if (reference_bgr8.size() != test_bgr8.size())
    {
        const int interpolation =
            (reference_bgr8.rows > test_bgr8.rows || reference_bgr8.cols > test_bgr8.cols)
                ? cv::INTER_AREA
                : cv::INTER_LINEAR;
        cv::resize(reference_bgr8, reference_resized, test_bgr8.size(), 0.0, 0.0, interpolation);
    }
    else
    {
        reference_resized = reference_bgr8;
    }

    // Compute absolute difference
    cv::Mat diff;
    cv::absdiff(reference_resized, test_bgr8, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    // Compute sum of squared errors across all channels
    const cv::Scalar sse_per_channel = cv::sum(diff);
    const double sse = sse_per_channel[0] + sse_per_channel[1] + sse_per_channel[2];
    
    // Check if images are identical
    if (sse <= 1e-10)
    {
        return std::numeric_limits<double>::infinity();
    }

    // Convert SSE to MSE and then to PSNR
    const double mse = sse / (3.0 * static_cast<double>(reference_resized.total()));
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}
