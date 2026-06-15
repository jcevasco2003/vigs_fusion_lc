#include <gtest/gtest.h>
#include <random>
#include <iostream>

#include "f_vigs_slam/MarginalizationFactor.hpp"
#include "reference/MarginalizationFactor_original.hpp"

using f_vigs_slam::MarginalizationFactor;
using vigs_fusion::MarginalizationInfo;

TEST(MarginalizationComparison, ResidualMatchesOriginal)
{
    // --- Crear infos paralelos ---
    f_vigs_slam::MarginalizationInfo info_new;
    vigs_fusion::MarginalizationInfo info_ref;

    // 4 bloques: P_prev[7], VB_prev[9], P_cur[7], VB_cur[9]
    info_new.keep_block_size = {7, 9, 7, 9};
    info_new.keep_block_idx  = {0, 1, 2, 3};
    info_new.sum_block_size  = 32;

    info_ref.keep_block_size = info_new.keep_block_size;
    info_ref.keep_block_idx  = info_new.keep_block_idx;
    info_ref.sum_block_size  = info_new.sum_block_size;

    info_new.keep_block_data.resize(4);
    info_ref.keep_block_data.resize(4);

    // Arrays de prueba
    double x0[32] = {
        1,2,3, 0,0,0,1,        // P_prev
        0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,  // VB_prev
        1.1,1.2,1.3, 0,0,0,1,  // P_cur
        0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19 // VB_cur
    };

    for (int i=0, offset=0; i<4; ++i) {
        int sz = info_new.keep_block_size[i];
        info_new.keep_block_data[i] = new double[sz];
        info_ref.keep_block_data[i] = new double[sz];
        std::copy(x0+offset, x0+offset+sz, info_new.keep_block_data[i]);
        std::copy(x0+offset, x0+offset+sz, info_ref.keep_block_data[i]);
        offset += sz;
    }

    // prior artificial
    Eigen::MatrixXd J = Eigen::MatrixXd::Random(6,6);
    Eigen::VectorXd r = Eigen::VectorXd::Random(6);

    info_new.linearized_jacobians = J;
    info_new.linearized_residuals = r;
    info_ref.linearized_jacobians = J;
    info_ref.linearized_residuals = r;

    // --- Crear factores ---
    f_vigs_slam::MarginalizationFactor factor_new(&info_new);
    vigs_fusion::MarginalizationFactor factor_ref(&info_ref);

    // --- Estado a evaluar ---
    double x[32];
    std::copy(x0, x0+32, x);
    const double* params[4] = {x, x+7, x+16, x+23}; // apuntar a cada bloque

    double r_new[6];
    double r_ref[6];

    EXPECT_TRUE(factor_new.Evaluate(params,r_new,nullptr));
    EXPECT_TRUE(factor_ref.Evaluate(params,r_ref,nullptr));

    Eigen::Map<Eigen::VectorXd> Rnew(r_new,6);
    Eigen::Map<Eigen::VectorXd> Rref(r_ref,6);

    EXPECT_LT((Rnew-Rref).norm(),1e-9);
}

TEST(MarginalizationComparison, JacobianMatchesOriginal)
{
    f_vigs_slam::MarginalizationInfo info_new;
    vigs_fusion::MarginalizationInfo info_ref;

    info_new.keep_block_size = {7, 9, 7, 9};
    info_new.keep_block_idx  = {0, 1, 2, 3};
    info_new.sum_block_size  = 32;

    info_ref.keep_block_size = info_new.keep_block_size;
    info_ref.keep_block_idx  = info_new.keep_block_idx;
    info_ref.sum_block_size  = info_new.sum_block_size;

    info_new.keep_block_data.resize(4);
    info_ref.keep_block_data.resize(4);

    double x0[32] = {0};
    for (int i=0, offset=0; i<4; ++i) {
        int sz = info_new.keep_block_size[i];
        info_new.keep_block_data[i] = new double[sz];
        info_ref.keep_block_data[i] = new double[sz];
        std::copy(x0+offset, x0+offset+sz, info_new.keep_block_data[i]);
        std::copy(x0+offset, x0+offset+sz, info_ref.keep_block_data[i]);
        offset += sz;
    }

    Eigen::MatrixXd J = Eigen::MatrixXd::Random(6,6);
    Eigen::VectorXd r = Eigen::VectorXd::Random(6);
    info_new.linearized_jacobians = J;
    info_new.linearized_residuals = r;
    info_ref.linearized_jacobians = J;
    info_ref.linearized_residuals = r;

    f_vigs_slam::MarginalizationFactor factor_new(&info_new);
    vigs_fusion::MarginalizationFactor factor_ref(&info_ref);

    double x[32] = {0};
    const double* params[4] = {x, x+7, x+16, x+23};

    double r_new[6], r_ref[6];
    double* Jnew_raw[1];
    double* Jref_raw[1];
    Jnew_raw[0] = new double[36];
    Jref_raw[0] = new double[36];

    factor_new.Evaluate(params,r_new,Jnew_raw);
    factor_ref.Evaluate(params,r_ref,Jref_raw);

    Eigen::Map<Eigen::Matrix<double,6,6,Eigen::RowMajor>> Jnew(Jnew_raw[0]);
    Eigen::Map<Eigen::Matrix<double,6,6,Eigen::RowMajor>> Jref(Jref_raw[0]);

    EXPECT_LT((Jnew-Jref).norm(),1e-9);
}