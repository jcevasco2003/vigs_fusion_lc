#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <memory>

#include "f_vigs_slam/Preintegration.hpp"

namespace f_vigs_slam
{
    class ImuCostFunction : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
    {
    public:
        explicit ImuCostFunction(std::shared_ptr<Preintegration> preintegration,
                                 double reprop_ba_thresh = 0.10,
                                 double reprop_bg_thresh = 0.01);
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    private:
        std::shared_ptr<Preintegration> pre_integration_;
        const Eigen::Vector3d G_{0.0, 0.0, 9.8};
        double reprop_ba_thresh_ = 0.10;
        double reprop_bg_thresh_ = 0.01;
    };
}