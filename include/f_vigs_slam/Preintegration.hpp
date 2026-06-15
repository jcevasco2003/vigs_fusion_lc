#pragma once

#include <Eigen/Dense>
#include <vector>

namespace f_vigs_slam
{
    /**
     * @brief Preintegration
     * Clase para preintegración de mediciones IMU entre frames consecutivos.
     * Basada en VIGS-Fusion y formulación de preintegración IMU estándar.
     */
    class Preintegration
    {
    public:
        Preintegration();

        /**
         * @brief Inicializa la preintegración con medición inicial y biases
         * @param _acc Aceleración inicial
         * @param _gyr Giroscopio inicial
         * @param _ba Bias del acelerómetro
         * @param _bg Bias del giroscopio
         * @param acc_n Ruido del acelerómetro (noise)
         * @param gyr_n Ruido del giroscopio (noise)
         * @param acc_w Random walk del acelerómetro
         * @param gyr_w Random walk del giroscopio
         */
        virtual void init(const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr,
                          const Eigen::Vector3d &_ba, const Eigen::Vector3d &_bg,
                          double acc_n, double gyr_n, double acc_w, double gyr_w);

        /**
         * @brief Agrega una medición IMU y propaga el estado
         * @param _dt Delta de tiempo desde última medición
         * @param _acc Aceleración medida
         * @param _gyr Giroscopio medido
         */
        void add_imu(double _dt, const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr);

        /**
         * @brief Propaga el estado con una medición IMU
         * @param _dt Delta de tiempo
         * @param _acc Aceleración medida
         * @param _gyr Giroscopio medido
         */
        virtual void propagate(double _dt, const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr);

        /**
         * @brief Re-propaga todas las mediciones con nuevos biases linealizados
         * @param _linearized_ba Nuevo bias del acelerómetro
         * @param _linearized_bg Nuevo bias del giroscopio
         */
        void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg);

        /**
         * @brief Evalúa el residual IMU para optimización
         * @return Residual de 15x1 [delta_p, delta_q, delta_v, delta_ba, delta_bg]
         */
        virtual Eigen::MatrixXd evaluate(
            const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
            const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
            const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
            const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj);

        /**
         * @brief Predice pose futura desde pose anterior usando preintegración
         * @param Pi Posición inicial
         * @param Qi Orientación inicial
         * @param Vi Velocidad inicial
         * @param Pj Posición predicha (output)
         * @param Qj Orientación predicha (output)
         * @param Vj Velocidad predicha (output)
         */
        void predict(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
                     Eigen::Vector3d &Pj, Eigen::Quaterniond &Qj, Eigen::Vector3d &Vj);

        // ===== DATOS PÚBLICOS =====
        const Eigen::Vector3d G = {0, 0, 9.81};  ///< Gravedad

        bool is_initialized;  ///< Flag de inicialización

        // Buffers de mediciones
        std::vector<double> dt_buf;
        std::vector<Eigen::Vector3d> acc_buf;
        std::vector<Eigen::Vector3d> gyr_buf;

        // Estado preintegrado
        double sum_dt;              ///< Suma de deltas de tiempo
        Eigen::Vector3d delta_p;    ///< Delta de posición
        Eigen::Quaterniond delta_q; ///< Delta de orientación
        Eigen::Vector3d delta_v;    ///< Delta de velocidad

        // Mediciones actuales
        Eigen::Vector3d acc, gyr;

        // Biases
        Eigen::Vector3d ba, bg;

        // Mediciones linealizadas
        Eigen::Vector3d linearized_acc, linearized_gyr;

        // Matrices para optimización
        Eigen::MatrixXd jacobian;   ///< Jacobiano (15x15)
        Eigen::MatrixXd covariance; ///< Covarianza (15x15)
        Eigen::MatrixXd noise;      ///< Matriz de ruido (18x18)
    };

} // namespace f_vigs_slam
