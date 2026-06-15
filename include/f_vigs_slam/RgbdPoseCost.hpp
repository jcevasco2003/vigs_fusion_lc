#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vector>

namespace f_vigs_slam
{
    // Forward declaration
    class GSSlam;

    /**
     * @brief Función de costo RGB-D para optimización de pose con Ceres
     * 
     * Esta función encapsula el error de renderización RGB-D entre:
     * - Imagen renderizada desde gaussianas 3D proyectadas a pose actual
     * - Imagen observada (RGB-D capturada)
     * 
     * Combina error fotométrico (color) y geométrico (profundidad).
     * 
     * **Estructura de residuales:**
     * - Dimensión: 6 (3 traslación + 3 rotación en espacio tangente)
     * - Parámetros: 1 bloque de 7D (pose IMU como [x,y,z,qx,qy,qz,qw])
     * 
     * **Implementación:**
     * - Calcula JtJ (6x6) y Jtr (6x1) mediante kernel GPU
     * - Regulariza con descomposición de eigenvalores (estabilidad numérica)
     * - Transforma de espacio cámara a espacio IMU
     * - Provee jacobiano analítico a Ceres
     * 
     * **Patrón de uso:**
     * ```cpp
     * RgbdPoseCost* visual_cost = new RgbdPoseCostFunction(gs_slam_ptr);
     * visual_cost->update(pyramid_level);  // Cambiar nivel de pirámide
     * problem.AddResidualBlock(visual_cost, nullptr, P_cur);
     * ```
     */
    class RgbdPoseCostFunction : public ceres::SizedCostFunction<6, 7>
    {
    public:
        /**
         * @brief Constructor
         * @param gs_slam Puntero a instancia de GSSlam (para acceso a renderización)
         */
        explicit RgbdPoseCostFunction(GSSlam* gs_slam);
        
        /**
         * @brief Destructor
         */
        ~RgbdPoseCostFunction();

        /**
         * @brief Actualiza el nivel de pirámide para optimización multi-escala
         * @param level Nivel de pirámide (0 = máxima resolución, N-1 = mínima)
         */
        void update(int level);

        /**
         * @brief Evalúa residuales y jacobianos (interfaz Ceres)
         * 
         * **Proceso:**
         * 1. Extrae pose IMU de parameters[0]
         * 2. Llama a gs_slam_->computeRgbdPoseJacobians() para obtener JtJ, Jtr
         * 3. Regulariza JtJ mediante eigenvalue decomposition
         * 4. Calcula pseudo-residuales: r* = sqrt(S_inv) * U^T * Jtr
         * 5. Calcula pseudo-jacobiano: J* = sqrt(S) * U^T
         * 6. Retorna r* y J* a Ceres
         * 
         * @param parameters Array de punteros a bloques de parámetros [P_imu]
         * @param residuals Output: vector de 6 residuales
         * @param jacobians Output: matriz 6×7 (si no es nullptr)
         * @return true si evaluación exitosa
         */
        bool Evaluate(double const* const* parameters,
                      double* residuals,
                      double** jacobians) const override;

        /**
         * @brief Versión no-const para permitir modificación interna
         * (Necesario porque kernels GPU modifican estado interno de GSSlam)
         */
        bool EvaluateNonConst(double const* const* parameters,
                              double* residuals,
                              double** jacobians);

    private:
        GSSlam* gs_slam_;  ///< Puntero a GSSlam (no posee memoria)
        int level_;        ///< Nivel actual de pirámide
    };

} // namespace f_vigs_slam
