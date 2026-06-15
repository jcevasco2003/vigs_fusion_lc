#pragma once

#include <vector>
#include <string>
#include <functional>
#include <random>
#include <stdexcept>
#include <cmath>

namespace f_vigs_slam
{
    struct KeyframeSelectionConfig
    {
        std::string method = "beta_binomial";
        float gumbel_alpha = 0.7f;
        float gumbel_beta = 2.0f;
        float beta_binomial_alpha = 0.7f;
        float beta_binomial_beta = 2.0f;
        float exponential_lambda = 1.0f;
        int sliding_window_window_size = -1;
    };

    /**
     * @class KeyframeSelector
     * @brief Selector genérico de keyframes para optimización con múltiples métodos
     * 
     * Permite cambiar dinámicamente entre diferentes estrategias de selección:
     * - "beta_binomial": Muestreo Beta-Binomial (exploración controlada)
     * - "uniform": Muestreo uniforme
     * - "exponential": Muestreo exponencial (bias hacia recientes)
     * - "sliding_window": Ventana deslizante
     * - "covisibility": Selección por covisibilidad (requiere argumentos externos)
     */
    class KeyframeSelector
    {
    public:
        KeyframeSelector(unsigned long seed = 0);
        ~KeyframeSelector();
        
        /**
         * @brief Muestrean índices de keyframes
         * 
         * @param n             Número de keyframes a seleccionar
         * @param total_kfs     Total de keyframes disponibles (índices: 0 a total_kfs-1)
         * @param config        Configuración encapsulada de muestreo
         * @return              Vector de índices seleccionados (sorted)
         */
        std::vector<int> sample(int n, int total_kfs,
                                const KeyframeSelectionConfig &config = KeyframeSelectionConfig());


        // ============================================================
        // Configuración
        // ============================================================
        
        void set_seed(unsigned long seed);
        void set_recent_bias_weight(float weight);  // Cómo de sesgado hacia recientes (default: 0.5)

    private:
        // ============================================================
        // Miembros
        // ============================================================
        
        std::mt19937 rng_;
        float recent_bias_weight_ = 0.5f;

        // ============================================================
        // Métodos de muestreo privados
        // ============================================================
        
        std::vector<int> sample_gumbel(int n, int total_kfs, const KeyframeSelectionConfig &config);
        std::vector<int> sample_beta_binomial(int n, int total_kfs, const KeyframeSelectionConfig &config);
        std::vector<int> sample_uniform(int n, int total_kfs, const KeyframeSelectionConfig &config);
        std::vector<int> sample_exponential(int n, int total_kfs, const KeyframeSelectionConfig &config);
        std::vector<int> sample_sliding_window(int n, int total_kfs, const KeyframeSelectionConfig &config);
        // ============================================================
        // Utilidades
        // ============================================================
        
        /**
         * @brief Muestrea de distribución Beta(alpha, beta)
         * Usa método de composición con Gamma(alpha) / (Gamma(alpha) + Gamma(beta))
         */
        float sample_beta(float alpha, float beta);
        
        /**
         * @brief Muestrea de distribución Gamma(shape, scale=1)
         * Usa transformación de exponencial
         */
        float sample_gamma(float shape);

        /**
         * @brief Muestrea n índices sin reemplazo del rango [0, total)
         */
        std::vector<int> sample_without_replacement(int n, int total);
    };

} // namespace f_vigs_slam