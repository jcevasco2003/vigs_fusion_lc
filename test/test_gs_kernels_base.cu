class GSSlamLikeTestBase : public ::testing::Test {
protected:

    // ============================================================
    // CONFIG
    // ============================================================
    int width  = 640;
    int height = 480;
    int nb_pyr_levels_ = 3;

    dim3 block = dim3(16,16);

    IntrinsicParameters intrinsics_;
    float depth_scale_ = 1.0f;

    // ============================================================
    // INPUTS CPU
    // ============================================================
    cv::Mat rgb_host_;    // CV_8UC3
    cv::Mat depth_host_;  // CV_32F

    // ============================================================
    // GPU DATA (igual que tu pipeline)
    // ============================================================
    cv::cuda::GpuMat rgb_gpu_;
    cv::cuda::GpuMat depth_gpu_;

    std::vector<cv::cuda::GpuMat> pyr_color_;
    std::vector<cv::cuda::GpuMat> pyr_depth_;
    std::vector<cv::cuda::GpuMat> pyr_normals_;
    std::vector<cv::cuda::GpuMat> pyr_dx_;
    std::vector<cv::cuda::GpuMat> pyr_dy_;

    // ============================================================
    // TEXTURAS (igual que vos)
    // ============================================================
    std::vector<cudaTextureObject_t> pyr_color_tex_;
    std::vector<cudaTextureObject_t> pyr_depth_tex_;
    std::vector<cudaTextureObject_t> pyr_normals_tex_;
    std::vector<cudaTextureObject_t> pyr_dx_tex_;
    std::vector<cudaTextureObject_t> pyr_dy_tex_;

    // ============================================================
    // SETUP
    // ============================================================
    void SetUp() override {
        initIntrinsics();
        generateInputs();
        initLikePipeline();
    }

    void TearDown() override {
        destroyTextureVector(pyr_color_tex_);
        destroyTextureVector(pyr_depth_tex_);
        destroyTextureVector(pyr_normals_tex_);
        destroyTextureVector(pyr_dx_tex_);
        destroyTextureVector(pyr_dy_tex_);

        cudaDeviceSynchronize();
    }

    // ============================================================
    // INTRINSICS
    // ============================================================
    void initIntrinsics() {
        intrinsics_.f = make_float2(525.f, 525.f);
        intrinsics_.c = make_float2(width/2.f, height/2.f);
    }

    // ============================================================
    // INPUTS GENERICOS
    // ============================================================
    void generateInputs() {
        rgb_host_ = cv::Mat(height, width, CV_8UC3);
        depth_host_ = cv::Mat(height, width, CV_32F, cv::Scalar(1.0f));

        std::mt19937 rng(0);

        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                rgb_host_.at<cv::Vec3b>(y,x) =
                    cv::Vec3b(rng()%255, rng()%255, rng()%255);
    }

    // ============================================================
    // PIPELINE (CLON DE initAndCopyImgs)
    // ============================================================
    void initLikePipeline() {

        // ---- NORMALIZACION ----
        cv::Mat rgb_bgr = rgb_host_;

        cv::Mat depth_float = depth_host_;

        // ---- UPLOAD ----
        rgb_gpu_.upload(rgb_bgr);
        depth_gpu_.upload(depth_float);

        pyr_color_.resize(nb_pyr_levels_);
        pyr_depth_.resize(nb_pyr_levels_);
        pyr_normals_.resize(nb_pyr_levels_);
        pyr_dx_.resize(nb_pyr_levels_);
        pyr_dy_.resize(nb_pyr_levels_);

        // nivel 0
        cv::cuda::cvtColor(rgb_gpu_, pyr_color_[0], cv::COLOR_BGR2BGRA);
        depth_gpu_.copyTo(pyr_depth_[0]);

        // ---- PIRAMIDE ----
        for (int i = 1; i < nb_pyr_levels_; i++) {
            cv::cuda::pyrDown(pyr_color_[i-1], pyr_color_[i]);
            cv::cuda::pyrDown(pyr_depth_[i-1], pyr_depth_[i]);
        }

        // ---- SOBEL ----
        for (int i = 0; i < nb_pyr_levels_; i++) {

            pyr_dx_[i].create(pyr_color_[i].size(), CV_32FC4);
            pyr_dy_[i].create(pyr_color_[i].size(), CV_32FC4);

            dim3 grid(
                (pyr_color_[i].cols + block.x - 1) / block.x,
                (pyr_color_[i].rows + block.y - 1) / block.y);

            size_t shared_bytes =
                (block.x + 2) * (block.y + 2) * sizeof(float4);

            computeSobelRgb_kernel<<<grid, block, shared_bytes>>>(
                reinterpret_cast<const uchar4*>(pyr_color_[i].ptr<uchar4>()),
                pyr_color_[i].step,
                reinterpret_cast<float4*>(pyr_dx_[i].ptr<float4>()),
                pyr_dx_[i].step,
                reinterpret_cast<float4*>(pyr_dy_[i].ptr<float4>()),
                pyr_dy_[i].step,
                pyr_color_[i].cols,
                pyr_color_[i].rows
            );

            checkCuda("computeSobelRgb");
        }

        // ---- NORMALES (solo nivel 0 como vos) ----
        pyr_normals_[0].create(pyr_depth_[0].size(), CV_32FC4);

        dim3 n_block(16,16);
        dim3 n_grid(
            (pyr_depth_[0].cols + n_block.x - 1) / n_block.x,
            (pyr_depth_[0].rows + n_block.y - 1) / n_block.y);

        computeNormalsFromDepth_kernel<<<n_grid, n_block>>>(
            createTextureObject<float>(pyr_depth_[0]), // temporal (opcional)
            pyr_normals_[0].ptr<float4>(),
            pyr_normals_[0].step,
            pyr_depth_[0].cols,
            pyr_depth_[0].rows,
            intrinsics_
        );

        checkCuda("computeNormals");

        // ---- TEXTURAS ----
        auto ensureSize = [&](auto& v) {
            if (v.size() != nb_pyr_levels_)
                v.resize(nb_pyr_levels_, 0);
        };

        ensureSize(pyr_color_tex_);
        ensureSize(pyr_depth_tex_);
        ensureSize(pyr_normals_tex_);
        ensureSize(pyr_dx_tex_);
        ensureSize(pyr_dy_tex_);

        for (int i = 0; i < nb_pyr_levels_; i++) {
            updateTexture<uchar4>(pyr_color_tex_[i],   pyr_color_[i]);
            updateTexture<float>(pyr_depth_tex_[i],    pyr_depth_[i]);
            updateTexture<float4>(pyr_normals_tex_[i], pyr_normals_[i]);
            updateTexture<float4>(pyr_dx_tex_[i],      pyr_dx_[i]);
            updateTexture<float4>(pyr_dy_tex_[i],      pyr_dy_[i]);
        }

        cudaDeviceSynchronize();
    }

    // ============================================================
    // HELPERS
    // ============================================================
    void checkCuda(const std::string& msg) {
        cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess)
            << "[CUDA ERROR] " << msg << " : "
            << cudaGetErrorString(err);
    }
};