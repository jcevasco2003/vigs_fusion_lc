#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <memory>
#include "Preintegration_original.hpp"

namespace vigs_fusion
{

    class ImuCostFunction : public ceres::SizedCostFunction<15, 7, 9, 7, 9> // résidu taille 15, 4 blocks de paramètres taille 7,9,7,9
    {
    protected:
        std::shared_ptr<Preintegration> pre_integration;
        const Eigen::Vector3d G{0, 0, 9.81};

    public:
        ImuCostFunction(std::shared_ptr<Preintegration> &_pre_integration);
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const; // ceres evaluate
    };

} // namespace vigs_fusion