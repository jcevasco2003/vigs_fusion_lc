#include "f_vigs_slam/PoseLocalParameterization.hpp"
#include "f_vigs_slam/LocalParameterization.hpp"

namespace f_vigs_slam
{
    // Implementacion alineada con VIGS-Fusion para mantener el mismo comportamiento
    // en la actualizacion de pose sobre manifold SE(3).

    bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        // Estado actual
        Eigen::Map<const Eigen::Vector3d> _p(x);
        Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

        // Perturbación local
        Eigen::Map<const Eigen::Vector3d> dp(delta);
        Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

        // Salida actualizada
        Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

        // Actualización
        p = _p + dp;
        q = (_q * dq).normalized();

        return true;
    }

    bool PoseLocalParameterization::ComputeJacobian(const double *, double *jacobian) const
    {
        // Jacobiano local 7x6
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        j.topRows<6>().setIdentity();
        j.bottomRows<1>().setZero();

        return true;
    }

    void PoseLocalParameterization::boxMinus(const double *xi, const double *xj,
                                             double *xi_minus_xj) const
    {
        // Diferencia entre poses en el espacio tangente
        Eigen::Map<const Eigen::Vector3d> pi(xi);
        Eigen::Map<const Eigen::Vector3d> pj(xj);

        const Eigen::Quaterniond qi(xi[6], xi[3], xi[4], xi[5]);
        const Eigen::Quaterniond qj(xj[6], xj[3], xj[4], xj[5]);

        Eigen::Map<Eigen::Vector3d> p(xi_minus_xj);
        p = pi - pj;

        xi_minus_xj[3] = 2.0 * (-qi.w() * qj.x() + qi.x() * qj.w() - qi.y() * qj.z() + qi.z() * qj.y());
        xi_minus_xj[4] = 2.0 * (-qi.w() * qj.y() + qi.x() * qj.z() + qi.y() * qj.w() - qi.z() * qj.x());
        xi_minus_xj[5] = 2.0 * (-qi.w() * qj.z() - qi.x() * qj.y() + qi.y() * qj.x() + qi.z() * qj.w());
    }

    Eigen::MatrixXd PoseLocalParameterization::boxMinusJacobianLeft(double const *, double const *xj) const
    {
        // Jacobiano analítico de boxMinus respecto del operando izquierdo
        const Eigen::Quaterniond qj(xj[6], xj[3], xj[4], xj[5]);

        Eigen::MatrixXd J(6, 7);
        J.setZero();
        J.block<3, 3>(0, 0).setIdentity();
        J.block<3, 4>(3, 3) << qj.w(), -qj.z(), qj.y(), -qj.x(),
            qj.z(), qj.w(), -qj.x(), -qj.y(),
            -qj.y(), qj.x(), qj.w(), -qj.z();
        J.block<3, 4>(3, 3) *= 2.;

        return J;
    }

    Eigen::MatrixXd PoseLocalParameterization::boxMinusJacobianRight(double const *xi, double const *) const
    {
        // Jacobiano analítico de boxMinus respecto del operando derecho
        const Eigen::Quaterniond qi(xi[6], xi[3], xi[4], xi[5]);

        Eigen::MatrixXd J(6, 7);
        J.setZero();
        J.block<3, 3>(0, 0).setIdentity();
        J.block<3, 4>(3, 3) << -qi.w(), qi.z(), -qi.y(), qi.x(),
            -qi.z(), -qi.w(), qi.x(), qi.y(),
            qi.y(), -qi.x(), -qi.w(), qi.z();
        J.block<3, 4>(3, 3) *= 2.;

        return J;
    }

    inline Eigen::Quaterniond PoseLocalParameterization::deltaQ(const Eigen::Vector3d &theta)
    {
        // Aproximación de primer orden de la exponencial de SO(3)
        Eigen::Quaterniond dq;
        Eigen::Vector3d half_theta = theta;
        half_theta /= 2.0;
        dq.w() = 1.0;
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

} // namespace f_vigs_slam
