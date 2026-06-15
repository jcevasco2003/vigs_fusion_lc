#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace f_vigs_slam
{
    /**
     * @brief Clase base para parametrizaciones locales en manifolds
     *
     * Hereda de ceres::LocalParameterization y expone las operaciones de
     * suma/resta en la variedad. Notemos que boxMinus es la resta a izquierda con nuestra notacion
     * - Suma a izquierda: (+)_L
     * - Resta a izquierda: (-)_L
     * - Suma a derecha: (+)_R
     * - Resta a derecha: (-)_R
     */
    class LocalParameterizationBase : public ceres::LocalParameterization
    {
    public:
        /**
         * @brief Calcula la resta en el espacio tangente
         * @param xi Parámetro del lado izquierdo
         * @param xj Parámetro del lado derecho
         * @param xi_minus_xj Resultado de (xi (-)_L xj)
         */
        virtual void boxMinus(const double *xi, const double *xj,
                              double *xi_minus_xj) const = 0;

        /**
         * @brief Derivada de la resta (-_L) respecto del parámetro izquierdo
         * @param xi Parámetro del lado izquierdo
         * @param xj Parámetro del lado derecho
         * @return Jacobiano de (xi (-)_L xj) respecto de xi
         */
        virtual Eigen::MatrixXd boxMinusJacobianLeft(double const *xi, double const *xj) const = 0;

        /**
         * @brief Derivada de la resta (-_L) respecto del parámetro derecho
         * @param xi Parámetro del lado izquierdo
         * @param xj Parámetro del lado derecho
         * @return Jacobiano de (xi (-)_L xj) respecto de xj
         */
        virtual Eigen::MatrixXd boxMinusJacobianRight(double const *xi, double const *xj) const = 0;
    };

} // namespace f_vigs_slam
