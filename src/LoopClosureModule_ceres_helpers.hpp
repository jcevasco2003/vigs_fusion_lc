#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "f_vigs_slam/PoseLocalParameterization.hpp"
#include "f_vigs_slam/RepresentationClasses.hpp"

namespace f_vigs_slam {

// Functor for AutoDiff: computes 6D residual (3 translational, 3 rotational angle-axis)
// Params: xi (7), xj (7) in layout [x,y,z, qx,qy,qz,qw]
// Measurement: rel_t (3), rel_q (qx,qy,qz,qw)
// sqrt_info: 6x6 upper triangular stored as Eigen::Matrix<double,6,6>
struct PoseGraphEdgeAutoDiff {
    PoseGraphEdgeAutoDiff(const Pose &meas, const Eigen::Matrix<double,6,6> &sqrt_info)
        : meas_t_{meas.position.x, meas.position.y, meas.position.z}, sqrt_info_(sqrt_info) {
        meas_q_ = Eigen::Quaterniond(meas.orientation.w, meas.orientation.x, meas.orientation.y, meas.orientation.z);
    }

    template <typename T>
    bool operator()(const T* const xi, const T* const xj, T* residual) const {
        // xi: [x,y,z, qx,qy,qz,qw]
        Eigen::Matrix<T,3,1> ti; ti << xi[0], xi[1], xi[2];
        Eigen::Quaternion<T> qi; qi.w() = xi[6]; qi.x() = xi[3]; qi.y() = xi[4]; qi.z() = xi[5];
        Eigen::Matrix<T,3,1> tj; tj << xj[0], xj[1], xj[2];
        Eigen::Quaternion<T> qj; qj.w() = xj[6]; qj.x() = xj[3]; qj.y() = xj[4]; qj.z() = xj[5];

        // predicted relative: T_pred = inv(Ti) * Tj
        Eigen::Quaternion<T> qi_conj = qi.conjugate();
        Eigen::Matrix<T,3,1> t_diff = tj - ti;
        Eigen::Matrix<T,3,1> t_pred = qi_conj * t_diff;
        Eigen::Quaternion<T> q_pred = qi_conj * qj;

        // measurement
        Eigen::Matrix<T,3,1> t_meas; t_meas << T(meas_t_[0]), T(meas_t_[1]), T(meas_t_[2]);
        Eigen::Quaternion<T> q_meas = Eigen::Quaternion<T>(T(meas_q_.w()), T(meas_q_.x()), T(meas_q_.y()), T(meas_q_.z()));

        // translation residual
        Eigen::Matrix<T,3,1> tres = t_pred - t_meas;

        // rotation residual: q_err = q_meas^{-1} * q_pred -> angle-axis
        Eigen::Quaternion<T> q_err = q_meas.conjugate() * q_pred;

        // ensure normalized
        q_err.normalize();
        if (q_err.w() < T(0)) {
            q_err.coeffs() = -q_err.coeffs(); // flip to ensure w>=0 for consistent angle-axis
        }

        T w = q_err.w();
        Eigen::Matrix<T,3,1> v(q_err.x(), q_err.y(), q_err.z());
        T vnorm = sqrt(v.squaredNorm());
        Eigen::Matrix<T,3,1> rres;
        const T eps = T(1e-8);
        T angle = T(2) * atan2(vnorm, w);
        if (vnorm > eps) {
            rres = (angle / vnorm) * v;
        } else {
            rres = Eigen::Matrix<T,3,1>::Zero();
        }

        // assemble raw residual (6)
        Eigen::Matrix<T,6,1> r_raw;
        r_raw.template block<3,1>(0,0) = tres;
        r_raw.template block<3,1>(3,0) = rres;

        // apply sqrt_info_: need to cast to T
        Eigen::Matrix<T,6,6> S;
        for (int i=0;i<6;++i) for (int j=0;j<6;++j) S(i,j) = T(sqrt_info_(i,j));
        Eigen::Matrix<T,6,1> r_final = S * r_raw;

        // PRINT DIAGNOSTICO, RECORDAR ELIMINAR
        if constexpr (std::is_same_v<T,double>)
            {
                double trans_norm =
                    std::sqrt(
                        double(tres[0]*tres[0] +
                            tres[1]*tres[1] +
                            tres[2]*tres[2]));

                double rot_norm =
                    std::sqrt(
                        double(rres[0]*rres[0] +
                            rres[1]*rres[1] +
                            rres[2]*rres[2]));

                double final_norm =
                    std::sqrt(
                        double(r_final.squaredNorm()));

                std::cout
                    << "[PGO][RESIDUAL]"
                    << " trans_norm=" << trans_norm
                    << " rot_norm=" << rot_norm
                    << " final_norm=" << final_norm
                    << std::endl;
            }
        
            static bool printed = false;

        if (!printed)
        {
            printed = true;

            std::cout << "\n=== LOOP RAW RESIDUAL ===\n";

            std::cout
                << ceres::JetOps<T>::GetScalar(r_raw(0))
                << " "
                << ceres::JetOps<T>::GetScalar(r_raw(1))
                << " "
                << ceres::JetOps<T>::GetScalar(r_raw(2))
                << " "
                << ceres::JetOps<T>::GetScalar(r_raw(3))
                << " "
                << ceres::JetOps<T>::GetScalar(r_raw(4))
                << " "
                << ceres::JetOps<T>::GetScalar(r_raw(5))
                << std::endl;
        }

        // TERMINAN DIAGNOSTICOS

        for (int k=0;k<6;++k) residual[k] = r_final(k,0);
        return true;
    }

private:
    double meas_t_[3];
    Eigen::Quaterniond meas_q_;
    Eigen::Matrix<double,6,6> sqrt_info_;
};

} // namespace f_vigs_slam
