#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "f_vigs_slam/LocalParameterization.hpp"

namespace f_vigs_slam
{
    /**
     * @brief Parametrización local para una pose SE(3) (posición + quaternion)
     *
     * Mapeo entre:
     * - Espacio global (7D): [x, y, z, qx, qy, qz, qw]
     * - Espacio tangente local (6D): [dx, dy, dz, wx, wy, wz]
     *
     * Este mapeo es necesario porque ceres optimiza en espacios sin restricciones,
     * y los quaternions deben mantenerse unitarios. Esto retira un grado de libertad.
     *
     * Usa el mapa exponencial del quaternion para perturbaciones de rotación.
     * GlobalSize() = 7, LocalSize() = 6
     */
    class PoseLocalParameterization : public LocalParameterizationBase
    {
    public:
        /**
         * @brief Operación de suma a izquierda: x_new = x_old (+)_L delta
         *
         * Posición: p_new = p_old + dp
         * Rotación: q_new = q_old (+) exp(ω) (normalizado)
         *
         * @param x Parámetro actual (7D)
         * @param delta Perturbación local (6D)
         * @param x_plus_delta Salida: x (+)_L delta (7D)
         */
        virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;

        /**
         * @brief Jacobiano de la suma (+)_L respecto a la perturbación local
         *
         * @param x Parámetro actual (7D)
         * @param jacobian Salida: J (7x6 row-major)
         */
        virtual bool ComputeJacobian(const double *x, double *jacobian) const;

        virtual int GlobalSize() const { return 7; }
        virtual int LocalSize() const { return 6; }

        /**
         * @brief Operación de resta a izquierda (inversa de (+)_L)
         * @param xi Operando izquierdo
         * @param xj Operando derecho
         * @param xi_minus_xj Salida: xi (-)_L xj (6D)
         */
        virtual void boxMinus(const double *xi, const double *xj,
                              double *xi_minus_xj) const;

        /**
         * @brief Jacobiano de (-)_L respecto del operando izquierdo
         */
        virtual Eigen::MatrixXd boxMinusJacobianLeft(double const *xi, double const *xj) const;

        /**
         * @brief Jacobiano de (-)_L respecto del operando derecho
         */
        virtual Eigen::MatrixXd boxMinusJacobianRight(double const *xi, double const *xj) const;

    private:
        /**
         * @brief Convierte un vector de rotación pequeño a quaternion (aprox. primer orden)
         *
         * ω = [wx, wy, wz] → q ≈ [1, wx/2, wy/2, wz/2] (no normalizado)
         */
        static Eigen::Quaterniond deltaQ(const Eigen::Vector3d &theta);
    };

} // namespace f_vigs_slam
