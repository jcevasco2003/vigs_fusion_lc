/*
#include <gtest/gtest.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include "f_vigs_slam/GSSlamTest.cuh"
#include "f_vigs_slam/RepresentationClasses.hpp"

// Helper para mostrar imágenes GPU
void showGpuImage(cv::cuda::GpuMat& imgGpu, const std::string& winName) {
    if (!imgGpu.empty()) {
        cv::Mat img;
        imgGpu.download(img);
        cv::imshow(winName, img);
        cv::waitKey(100); // esperar 100ms para ver la imagen
    }
}

// Helper para crear gaussianas dummy
f_vigs_slam::Gaussians createDummyGaussians(int n) {
    using Gaussians = f_vigs_slam::Gaussians;
    Gaussians g;
    g.positions.resize(n, Eigen::Vector3f::Zero());
    g.scales.resize(n, Eigen::Vector3f::Ones());
    g.orientations.resize(n, Eigen::Vector4f::Zero());
    g.colors.resize(n, cv::Vec3f(1.0f, 0.0f, 0.0f));
    g.opacities.resize(n, 1.0f);
    return g;
}

// Helper para crear patrones (columna, diagonal, cuadricula)
f_vigs_slam::Gaussians createPatternGaussians(int w, int h) {
    using Gaussians = f_vigs_slam::Gaussians;
    Gaussians g;
    int n = w * h;
    g.positions.resize(n);
    g.scales.resize(n, Eigen::Vector3f::Ones());
    g.orientations.resize(n, Eigen::Vector4f::Zero());
    g.colors.resize(n, cv::Vec3f(1.0f, 0.0f, 0.0f));
    g.opacities.resize(n, 1.0f);

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            g.positions[y * w + x] = Eigen::Vector3f(x, y, 0);

    return g;
}

// Fixture
class RasterizationDisplayTest : public ::testing::Test {
protected:
    void SetUp() override {
        test = std::make_unique<f_vigs_slam::GSSlamTest>(slam);

        test->nGaussians() = 0;
        test->maxGaussians() = 1024;
        test->tileSize() = make_uint2(16, 16);
        test->gaussians() = createDummyGaussians(test->maxGaussians());

        camera_pose = f_vigs_slam::CameraPose(); // identidad
        intrinsics = f_vigs_slam::IntrinsicParameters();
        width = 640;
        height = 480;
    }

    f_vigs_slam::GSSlam slam;
    std::unique_ptr<f_vigs_slam::GSSlamTest> test;
    f_vigs_slam::CameraPose camera_pose;
    f_vigs_slam::IntrinsicParameters intrinsics;
    int width, height;
};

// ====================== TEST: Columna de gaussianas ======================
TEST_F(RasterizationDisplayTest, SingleColumnDisplay) {
    test->gaussians() = createPatternGaussians(1, height);
    test->nGaussians() = height;

    slam.prepareRasterization(camera_pose, intrinsics, width, height);
    slam.rasterize(camera_pose, intrinsics, width, height);

    showGpuImage(test->renderedRgbGpu(), "SingleColumn_RGB");
    showGpuImage(test->renderedDepthGpu(), "SingleColumn_Depth");

    EXPECT_GT(test->last_nb_instances_(), 0);
}

// ====================== TEST: Diagonal de gaussianas ======================
TEST_F(RasterizationDisplayTest, DiagonalDisplay) {
    int n = std::min(width, height);
    #using Gaussians = f_vigs_slam::Gaussians;
    Gaussians g;
    g.positions.resize(n);
    g.scales.resize(n, Eigen::Vector3f::Ones());
    g.orientations.resize(n, Eigen::Vector4f::Zero());
    g.colors.resize(n, cv::Vec3f(0.0f, 1.0f, 0.0f));
    g.opacities.resize(n, 1.0f);

    for (int i = 0; i < n; i++)
        g.positions[i] = Eigen::Vector3f(i, i, 0);

    test->gaussians() = g;
    test->nGaussians() = n;

    slam.prepareRasterization(camera_pose, intrinsics, width, height);
    slam.rasterize(camera_pose, intrinsics, width, height);

    showGpuImage(test->renderedRgbGpu(), "Diagonal_RGB");
    showGpuImage(test->renderedDepthGpu(), "Diagonal_Depth");

    EXPECT_GT(test->last_nb_instances_(), 0);
}

// ====================== TEST: Cuadricula 4x4 ======================
TEST_F(RasterizationDisplayTest, RenderViewPatternDisplay) {
    test->gaussians() = createPatternGaussians(4,4);
    test->nGaussians() = 16;

    cv::cuda::GpuMat rgb, depth;
    bool ok = slam.renderView(camera_pose, intrinsics, width, height, rgb, depth);

    EXPECT_TRUE(ok);
    EXPECT_EQ(rgb.cols, width);
    EXPECT_EQ(rgb.rows, height);
    EXPECT_EQ(depth.cols, width);
    EXPECT_EQ(depth.rows, height);

    showGpuImage(rgb, "4x4Pattern_RGB");
    showGpuImage(depth, "4x4Pattern_Depth");
}

// ====================== TEST: No gaussianas ======================
TEST_F(RasterizationDisplayTest, NoGaussians) {
    test->nGaussians() = 0;

    slam.prepareRasterization(camera_pose, intrinsics, width, height);
    EXPECT_EQ(test->last_nb_instances_(), 0);

    slam.rasterize(camera_pose, intrinsics, width, height);
    EXPECT_TRUE(test->renderedRgbGpu().empty() || test->renderedDepthGpu().empty());
}

// ====================== TEST: Dimensiones inválidas ======================
TEST_F(RasterizationDisplayTest, InvalidDimensions) {
    test->nGaussians() = 10;

    slam.prepareRasterization(camera_pose, intrinsics, -1, 480);
    EXPECT_EQ(test->last_nb_instances_(), 0);

    slam.prepareRasterization(camera_pose, intrinsics, 640, 0);
    EXPECT_EQ(test->last_nb_instances_(), 0);
}