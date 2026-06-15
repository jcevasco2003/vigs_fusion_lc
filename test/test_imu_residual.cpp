#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <random>

#include "f_vigs_slam/Preintegration.hpp"
#include "f_vigs_slam/ImuCostFunction.hpp"

#include "reference/Preintegration_original.hpp"
#include "reference/ImuCostFunction_original.hpp"

TEST(ImuPreintegrationTest, ResidualMatchesReference)
{
    Eigen::Vector3d acc(0.1, -0.2, 9.7);
    Eigen::Vector3d gyr(0.01, 0.02, -0.01);

    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    double dt = 0.01;

    auto my_preint  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref_preint = std::make_shared<vigs_fusion::Preintegration>();

    my_preint->init(acc, gyr, ba, bg, 0.1,0.1,0.1,0.1);
    ref_preint->init(acc, gyr, ba, bg, 0.1,0.1,0.1,0.1);

    for(int i=0;i<100;i++){
        my_preint->add_imu(dt, acc, gyr);
        ref_preint->add_imu(dt, acc, gyr);
    }

    Eigen::Vector3d Pi(0,0,0);
    Eigen::Vector3d Vi(0,0,0);
    Eigen::Quaterniond Qi = Eigen::Quaterniond::Identity();

    Eigen::Vector3d Pj(0.1,0.1,0.2);
    Eigen::Vector3d Vj(0.01,0.01,0.02);
    Eigen::Quaterniond Qj = Eigen::Quaterniond::Identity();

    auto r_my = my_preint->evaluate(Pi,Qi,Vi,ba,bg,
                                    Pj,Qj,Vj,ba,bg);

    auto r_ref = ref_preint->evaluate(Pi,Qi,Vi,ba,bg,
                                      Pj,Qj,Vj,ba,bg);

    double diff = (r_my - r_ref).norm();

    EXPECT_LT(diff, 1e-6);
}

TEST(ImuPreintegrationTest, DeltaStateMatchesReference)
{
    Eigen::Vector3d acc(0.3, -0.1, 9.6);
    Eigen::Vector3d gyr(0.02, -0.01, 0.03);
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();

    auto my_preint  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref_preint = std::make_shared<vigs_fusion::Preintegration>();

    my_preint->init(acc, gyr, ba, bg, 0.1,0.1,0.1,0.1);
    ref_preint->init(acc, gyr, ba, bg, 0.1,0.1,0.1,0.1);

    for(int i=0;i<200;i++){
        my_preint->add_imu(0.005, acc, gyr);
        ref_preint->add_imu(0.005, acc, gyr);
    }

    EXPECT_LT((my_preint->delta_p - ref_preint->delta_p).norm(), 1e-6);
    EXPECT_LT((my_preint->delta_v - ref_preint->delta_v).norm(), 1e-6);

    Eigen::Quaterniond dq_err =
        my_preint->delta_q.inverse() * ref_preint->delta_q;

    EXPECT_LT(dq_err.vec().norm(), 1e-6);
}

TEST(ImuPreintegrationTest, RepropagationMatchesReference)
{
    Eigen::Vector3d acc(0.1,0.2,9.7);
    Eigen::Vector3d gyr(0.01,-0.02,0.005);

    Eigen::Vector3d ba0(0.01,0.02,-0.01);
    Eigen::Vector3d bg0(0.001,-0.002,0.003);

    auto my  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref = std::make_shared<vigs_fusion::Preintegration>();

    my->init(acc,gyr,ba0,bg0,0.1,0.1,0.1,0.1);
    ref->init(acc,gyr,ba0,bg0,0.1,0.1,0.1,0.1);

    for(int i=0;i<100;i++){
        my->add_imu(0.01,acc,gyr);
        ref->add_imu(0.01,acc,gyr);
    }

    Eigen::Vector3d ba_new(0.02,-0.01,0.005);
    Eigen::Vector3d bg_new(-0.002,0.003,0.001);

    my->repropagate(ba_new,bg_new);
    ref->repropagate(ba_new,bg_new);

    EXPECT_LT((my->delta_p - ref->delta_p).norm(), 1e-6);
    EXPECT_LT((my->delta_v - ref->delta_v).norm(), 1e-6);

    Eigen::Quaterniond dq_err =
        my->delta_q.inverse() * ref->delta_q;

    EXPECT_LT(dq_err.vec().norm(), 1e-6);
}

TEST(ImuPreintegrationTest, PredictMatchesReference)
{
    Eigen::Vector3d acc(0.1,0.1,9.6);
    Eigen::Vector3d gyr(0.01,0.01,-0.01);

    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();

    auto my  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref = std::make_shared<vigs_fusion::Preintegration>();

    my->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);
    ref->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);

    for(int i=0;i<100;i++){
        my->add_imu(0.01,acc,gyr);
        ref->add_imu(0.01,acc,gyr);
    }

    Eigen::Vector3d Pi(0,0,0), Vi(0,0,0);
    Eigen::Quaterniond Qi = Eigen::Quaterniond::Identity();

    Eigen::Vector3d Pj_my,Vj_my;
    Eigen::Quaterniond Qj_my;

    Eigen::Vector3d Pj_ref,Vj_ref;
    Eigen::Quaterniond Qj_ref;

    my->predict(Pi,Qi,Vi,Pj_my,Qj_my,Vj_my);
    ref->predict(Pi,Qi,Vi,Pj_ref,Qj_ref,Vj_ref);

    EXPECT_LT((Pj_my-Pj_ref).norm(),1e-6);
    EXPECT_LT((Vj_my-Vj_ref).norm(),1e-6);

    Eigen::Quaterniond dq = Qj_my.inverse()*Qj_ref;
    EXPECT_LT(dq.vec().norm(),1e-6);
}

TEST(ImuPreintegrationTest, CovarianceMatchesReference)
{
    Eigen::Vector3d acc(0.2,-0.1,9.8);
    Eigen::Vector3d gyr(0.02,0.01,-0.01);

    auto my  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref = std::make_shared<vigs_fusion::Preintegration>();

    my->init(acc,gyr,Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),0.1,0.1,0.1,0.1);
    ref->init(acc,gyr,Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),0.1,0.1,0.1,0.1);

    for(int i=0;i<200;i++){
        my->add_imu(0.01,acc,gyr);
        ref->add_imu(0.01,acc,gyr);
    }

    EXPECT_LT((my->covariance - ref->covariance).norm(),1e-6);
}

TEST(ImuPreintegrationTest, NonIdentityInitialState)
{
    Eigen::Vector3d acc(0.2,-0.1,9.7);
    Eigen::Vector3d gyr(0.02,0.01,-0.03);

    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();

    auto my  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref = std::make_shared<vigs_fusion::Preintegration>();

    my->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);
    ref->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);

    for(int i=0;i<300;i++){
        my->add_imu(0.005,acc,gyr);
        ref->add_imu(0.005,acc,gyr);
    }

    Eigen::Vector3d Pi(1.2,-0.7,0.3);
    Eigen::Vector3d Vi(0.4,-0.2,0.1);

    Eigen::Quaterniond Qi =
        Eigen::AngleAxisd(0.3,Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(-0.4,Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.2,Eigen::Vector3d::UnitZ());

    Eigen::Vector3d Pj(1.5,-0.6,0.8);
    Eigen::Vector3d Vj(0.5,-0.1,0.2);
    Eigen::Quaterniond Qj =
        Eigen::AngleAxisd(-0.1,Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(0.2,Eigen::Vector3d::UnitY());

    auto r_my = my->evaluate(Pi,Qi,Vi,ba,bg,Pj,Qj,Vj,ba,bg);
    auto r_ref = ref->evaluate(Pi,Qi,Vi,ba,bg,Pj,Qj,Vj,ba,bg);

    EXPECT_LT((r_my-r_ref).norm(),1e-6);
}

TEST(ImuPreintegrationTest, LongIntegrationStability)
{
    Eigen::Vector3d acc(0.3,-0.2,9.6);
    Eigen::Vector3d gyr(0.01,0.02,-0.015);

    auto my  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref = std::make_shared<vigs_fusion::Preintegration>();

    my->init(acc,gyr,Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),0.1,0.1,0.1,0.1);
    ref->init(acc,gyr,Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),0.1,0.1,0.1,0.1);

    for(int i=0;i<5000;i++){
        my->add_imu(0.002,acc,gyr);
        ref->add_imu(0.002,acc,gyr);
    }

    EXPECT_LT((my->delta_p - ref->delta_p).norm(),1e-6);
    EXPECT_LT((my->delta_v - ref->delta_v).norm(),1e-6);

    Eigen::Quaterniond dq =
        my->delta_q.inverse()*ref->delta_q;

    EXPECT_LT(dq.vec().norm(),1e-6);
}

TEST(ImuPreintegrationTest, MonteCarloConsistency)
{
    std::mt19937 rng(0);
    std::normal_distribution<double> n(0.0,1.0);

    for(int trial=0; trial<50; trial++)
    {
        Eigen::Vector3d acc(0.2+n(rng)*0.05,
                            -0.1+n(rng)*0.05,
                            9.7+n(rng)*0.05);

        Eigen::Vector3d gyr(0.01+n(rng)*0.01,
                            0.02+n(rng)*0.01,
                            -0.01+n(rng)*0.01);

        Eigen::Vector3d ba(n(rng)*0.01,n(rng)*0.01,n(rng)*0.01);
        Eigen::Vector3d bg(n(rng)*0.001,n(rng)*0.001,n(rng)*0.001);

        auto my  = std::make_shared<f_vigs_slam::Preintegration>();
        auto ref = std::make_shared<vigs_fusion::Preintegration>();

        my->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);
        ref->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);

        for(int i=0;i<200;i++)
            my->add_imu(0.01,acc,gyr),
            ref->add_imu(0.01,acc,gyr);

        EXPECT_LT((my->delta_p - ref->delta_p).norm(),1e-6);
        EXPECT_LT((my->delta_v - ref->delta_v).norm(),1e-6);

        Eigen::Quaterniond dq =
            my->delta_q.inverse()*ref->delta_q;

        EXPECT_LT(dq.vec().norm(),1e-6);
    }
}

TEST(ImuPreintegrationTest, LargeBiasConsistency)
{
    Eigen::Vector3d acc(0.3,-0.2,9.5);
    Eigen::Vector3d gyr(0.05,-0.04,0.03);

    // Bias grandes (caso difícil)
    Eigen::Vector3d ba(0.4,-0.3,0.2);
    Eigen::Vector3d bg(0.05,-0.06,0.04);

    auto my  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref = std::make_shared<vigs_fusion::Preintegration>();

    my->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);
    ref->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);

    for(int i=0;i<300;i++){
        my->add_imu(0.01,acc,gyr);
        ref->add_imu(0.01,acc,gyr);
    }

    EXPECT_LT((my->delta_p - ref->delta_p).norm(),1e-6);
    EXPECT_LT((my->delta_v - ref->delta_v).norm(),1e-6);

    Eigen::Quaterniond dq = my->delta_q.inverse()*ref->delta_q;
    EXPECT_LT(dq.vec().norm(),1e-6);

    EXPECT_LT((my->covariance - ref->covariance).norm(),1e-6);
}

TEST(ImuPreintegrationTest, VariableDtConsistency)
{
    Eigen::Vector3d acc(0.15,0.05,9.7);
    Eigen::Vector3d gyr(0.02,-0.01,0.015);

    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();

    auto my  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref = std::make_shared<vigs_fusion::Preintegration>();

    my->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);
    ref->init(acc,gyr,ba,bg,0.1,0.1,0.1,0.1);

    // dt variable simulando timestamps reales
    std::vector<double> dts = {
        0.008,0.011,0.009,0.012,0.010,
        0.007,0.013,0.010,0.009,0.012
    };

    for(int k=0;k<40;k++){
        for(double dt : dts){
            my->add_imu(dt,acc,gyr);
            ref->add_imu(dt,acc,gyr);
        }
    }

    EXPECT_LT((my->delta_p - ref->delta_p).norm(),1e-6);
    EXPECT_LT((my->delta_v - ref->delta_v).norm(),1e-6);

    Eigen::Quaterniond dq = my->delta_q.inverse()*ref->delta_q;
    EXPECT_LT(dq.vec().norm(),1e-6);

    EXPECT_LT((my->covariance - ref->covariance).norm(),1e-6);
}

TEST(ImuPreintegrationTest, LargeRotationMotion)
{
    Eigen::Vector3d acc(0.0, 0.0, 9.81);
    Eigen::Vector3d gyr(0.5, -0.3, 0.2);

    auto my  = std::make_shared<f_vigs_slam::Preintegration>();
    auto ref = std::make_shared<vigs_fusion::Preintegration>();

    my->init(acc,gyr,Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),0.1,0.1,0.1,0.1);
    ref->init(acc,gyr,Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),0.1,0.1,0.1,0.1);

    for(int i=0;i<500;i++){
        my->add_imu(0.01,acc,gyr);
        ref->add_imu(0.01,acc,gyr);
    }

    Eigen::Quaterniond dq =
        my->delta_q.inverse()*ref->delta_q;

    EXPECT_LT(dq.vec().norm(),1e-6);
}