#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <pthread.h>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <cstring>

// REPASAR

namespace f_vigs_slam
{

/**
 * @brief Información de un bloque residual para marginalización
 * 
 * Encapsula el factor de costo de marginalización con sus parámetros
 * y marca cuáles variables deben eliminarse del problema (drop_set).
 * 
 * Durante la evaluación, calcula residuales y jacobianos, aplicando
 * opcionalmente una función de pérdida robusta.
 */
struct ResidualBlockInfo
{
    /**
     * @brief Constructor
     * 
     * @param _cost_function Función de costo de Ceres
     * @param _loss_function Función de pérdida robusta (nullptr = sin robustez)
     * @param _parameter_blocks Punteros a bloques de parámetros del costo
     * @param _drop_set Índices de parámetros a marginalizar (eliminar)
     */
    ResidualBlockInfo(ceres::CostFunction *_cost_function,
                      ceres::LossFunction *_loss_function,
                      std::vector<double *> _parameter_blocks,
                      std::vector<int> _drop_set);

    /**
     * @brief Evalúa residuales y jacobianos del factor de costo
     * 
     * Si hay loss_function, aplica robustez mediante escalado adaptativo
     * de residuales y jacobianos según el método de Ceres.
     */
    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    /**
     * @brief Convierte tamaño global a local (manifold dimension)
     * 
     * Para poses SE(3): 7D global (posición + quaternion) → 6D local (tangente)
     */
    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

/**
 * @brief Estructura para paralelizar construcción de matriz de información
 * 
 * Cada thread procesa un subconjunto de factores y acumula su contribución
 * a la matriz A = J^T*J y vector b = J^T*r.
 */
struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; // Tamaño global
    std::unordered_map<long, int> parameter_block_idx;  // Índice en vector de estado
};

/**
 * @brief Gestiona el proceso completo de marginalización
 * 
 * Implementa el complemento de Schur para eliminar variables antiguas del
 * factor graph manteniendo su información como prior linearizado.
 * 
 * **Flujo de uso:**
 * 1. Crear MarginalizationInfo
 * 2. addResidualBlockInfo() para cada factor (IMU, visual, prior anterior)
 * 3. preMarginalize(): evalúa todos los factores en el punto de linearización
 * 4. marginalize(): construye A/b, aplica complemento de Schur, descompone prior
 * 5. getParameterBlocks(): obtiene bloques a mantener para siguiente iteración
 * 
 */
class MarginalizationInfo
{
public:
    MarginalizationInfo() : valid(true) {}
    ~MarginalizationInfo();

    /**
     * @brief Convierte tamaño global a local (manifold)
     */
    int localSize(int size) const;

    /**
     * @brief Convierte tamaño local a global
     */
    int globalSize(int size) const;

    /**
     * @brief Añade un bloque residual al sistema de marginalización
     * 
     * Registra el factor, sus parámetros y marca cuáles eliminar (drop_set).
     */
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);

    /**
     * @brief Inicializa índices y matrices para marginalización
     * 
     * Asigna índices a variables (primero a marginalizar [0,m), luego a mantener [m,m+n)).
     * Reserva memoria para A (matriz de información) y b (vector residual ponderado).
     */
    void init();

    /**
     * @brief Evalúa todos los factores y copia punto de linearización
     * 
     * Llama Evaluate() en cada ResidualBlockInfo y guarda valores actuales
     * de parámetros en parameter_block_data (punto donde se lineariza).
     */
    void preMarginalize();

    /**
     * @brief Ejecuta marginalización mediante complemento de Schur
     * 
     * 1. Construye A = J^T*J y b = J^T*r usando threads
     * 2. Particiona A en bloques [A_mm, A_mr; A_rm, A_rr]
     * 3. Calcula A* = A_rr - A_rm * A_mm^{-1} * A_mr (complemento de Schur)
     * 4. Descompone A* = J^T*J mediante eigendecomposición
     * 5. Guarda J y r como prior linearizado para siguiente optimización
     */
    void marginalize();

    /**
     * @brief Obtiene bloques de parámetros a mantener (no marginalizados)
     * 
     * @param addr_shift Mapeo de direcciones viejas a nuevas tras actualización de estado
     * @return Vector de punteros a bloques conservados
     */
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;           // Factores a marginalizar
    int m, n;                                            // m: vars a eliminar, n: vars a mantener
    std::unordered_map<long, int> parameter_block_size; // Tamaño global (7 para poses)
    int sum_block_size;                                  // Suma de tamaños de bloques conservados
    std::unordered_map<long, int> parameter_block_idx;  // Índice en vector de estado
    std::unordered_map<long, double *> parameter_block_data; // Punto de linearización

    std::vector<int> keep_block_size;      // Tamaños de bloques conservados
    std::vector<int> keep_block_idx;       // Índices de bloques conservados
    std::vector<double *> keep_block_data; // Valores de linearización conservados

    Eigen::MatrixXd linearized_jacobians; // J: jacobiano del prior marginalizado (n×n)
    Eigen::VectorXd linearized_residuals; // r: residual del prior (n×1)
    const double eps = 1e-8;              // Umbral para pseudo-inversión
    bool valid;                           // false si m=0 (no hay qué marginalizar)

    Eigen::MatrixXd A; // Matriz de información completa (m+n × m+n)
    Eigen::VectorXd b; // Vector residual ponderado (m+n)
};

/**
 * @brief Factor de costo de Ceres que aplica el prior marginalizado
 * 
 * Usa jacobiano y residual linearizados de MarginalizationInfo para imponer
 * restricciones de keyframes marginalizados en optimizaciones futuras.
 * 
 * **Residual:** r = r_0 + J * (x - x_0)
 * - r_0: residual linearizado en punto de marginalización
 * - J: jacobiano linearizado
 * - x: estado actual de bloques conservados
 * - x_0: punto de linearización (guardado en keep_block_data)
 * 
 * **Tratamiento de quaterniones:**
 * Para poses (tamaño 7), la diferencia x - x_0 usa la parte vectorial del
 * quaternion relativo: 2 * (q_0^{-1} ⊗ q).vec()
 */
class MarginalizationFactor : public ceres::CostFunction
{
public:
    MarginalizationFactor();

    /**
     * @brief Constructor con información de marginalización
     * 
     * @param _marginalization_info Resultado de marginalize() en iteración previa
     */
    MarginalizationFactor(MarginalizationInfo *_marginalization_info);

    /**
     * @brief Inicializa dimensiones del factor desde MarginalizationInfo
     * 
     * Configura num_residuals y parameter_block_sizes según bloques conservados.
     */
    void init(MarginalizationInfo *_marginalization_info);

    /**
     * @brief Evalúa residual y jacobianos del prior marginalizado
     * 
     * @param parameters Bloques de parámetros actuales (estado optimizado)
     * @param residuals Salida: r = r_0 + J * (x - x_0)
     * @param jacobians Salida: J (jacobianos respecto a cada bloque)
     * @return true si evaluación exitosa
     * 
     * Calcula diferencia x - x_0 con geometría correcta (lineal para posición,
     * quaternion relativo para orientación), aplica jacobiano linearizado.
     */
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo *marginalization_info;
};

} // namespace f_vigs_slam
