#include <f_vigs_slam/KeyframeSelector.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

namespace f_vigs_slam
{

    KeyframeSelector::KeyframeSelector(unsigned long seed)
        : rng_(seed == 0 ? std::random_device{}() : seed)
    {
    }

    KeyframeSelector::~KeyframeSelector() = default;

    std::vector<int> KeyframeSelector::sample(int n, int total_kfs,
                                              const KeyframeSelectionConfig &config)
    {
        if (n <= 0 || total_kfs <= 0) {
            return {};
        }
        n = std::min(n, total_kfs);

        if (config.method == "beta_binomial") {
            return sample_beta_binomial(n, total_kfs, config);
        }
        else if (config.method == "gumbel") {
            return sample_gumbel(n, total_kfs, config);
        }
        else if (config.method == "uniform") {
            return sample_uniform(n, total_kfs, config);
        }
        else if (config.method == "exponential") {
            return sample_exponential(n, total_kfs, config);
        }
        else if (config.method == "sliding_window") {
            return sample_sliding_window(n, total_kfs, config);
        }
        else {
            throw std::invalid_argument("KeyframeSelector metodo desconocido: " + config.method);
        }
    }

    void KeyframeSelector::set_seed(unsigned long seed)
    {
        rng_.seed(seed == 0 ? std::random_device{}() : seed);
    }

    
    // ============================================================================
    // Método -: Gumbel
    // ============================================================================

    std::vector<int> KeyframeSelector::sample_gumbel(int n, int total_kfs,
                                                     const KeyframeSelectionConfig &config)
    {
        const float alpha = config.gumbel_alpha;
        const float beta = config.gumbel_beta;

        // El algoritmo Beta-Binomial:
        // 1. Muestreamos p ~ Beta(alpha, beta)
        // 2. Para cada índice k: probabilidad p_k = Binomial(total_kfs-1, p, k)
        // 3. Muestreamos sin reemplazo según estas probabilidades

        std::vector<float> probabilities(total_kfs, 0.0f);

        // Muestreamos p ~ Beta(alpha, beta)
        float p = sample_beta(alpha, beta);

        // Probabilidades binomiales para cada índice
        // P(k) proporcional a: C(n,k) * p^k * (1-p)^(n-k)
        // Donde n = total_kfs - 1 (total de "ensayos")
        
        // Calculamos en escala logarítmica para evitar underflow
        std::vector<float> log_probs(total_kfs, 0.0f);
        float log_p = std::log(p + 1e-10f);
        float log_1_minus_p = std::log(1.0f - p + 1e-10f);

        const int n_trials = total_kfs - 1;
        for (int k = 0; k < total_kfs; k++) {
            // log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
            float logC = static_cast<float>(
                std::lgamma(n_trials + 1.0f) - std::lgamma(k + 1.0f) - std::lgamma(n_trials - k + 1.0f)
            );
            log_probs[k] = logC + k * log_p + (n_trials - k) * log_1_minus_p;
        }

        // Convertimos a probabilidades normalizadas
        float max_log_prob = *std::max_element(log_probs.begin(), log_probs.end());
        float sum_exp = 0.0f;
        
        for (int k = 0; k < total_kfs; k++) {
            probabilities[k] = std::exp(log_probs[k] - max_log_prob);
            sum_exp += probabilities[k];
        }

        for (int k = 0; k < total_kfs; k++) {
            probabilities[k] /= sum_exp;
        }

        // Muestreamos sin reemplazo usando muestreo por Gumbel-Max trick (aproximado)
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::vector<std::pair<float, int>> scored_indices;

        for (int k = 0; k < total_kfs; k++) {
            // Gumbel-Max: score = log(p) + (-log(-log(uniform(0,1))))
            float u = uniform(rng_);
            float gumbel = -std::log(-std::log(u + 1e-10f) + 1e-10f);
            float score = std::log(probabilities[k] + 1e-10f) + gumbel;
            scored_indices.push_back({score, k});
        }

        // Ordenamos por score descendente y tomar los top n
        std::partial_sort(scored_indices.begin(), 
                         scored_indices.begin() + n,
                         scored_indices.end(),
                         [](const auto &a, const auto &b) { return a.first > b.first; });

        std::vector<int> result;
        for (int i = 0; i < n; i++) {
            result.push_back(scored_indices[i].second);
        }

        std::sort(result.begin(), result.end());
        return result;
    }

    

    // ============================================================================
    // Método 1: Beta-Binomial
    // ============================================================================

    std::vector<int> KeyframeSelector::sample_beta_binomial(int n, int total_kfs,
                                                             const KeyframeSelectionConfig &config)
    {
        const float alpha = config.beta_binomial_alpha;
        const float beta = config.beta_binomial_beta;

        // 1) Muestreamos p ~ Beta(alpha, beta)
        float p = sample_beta(alpha, beta);

        // 2) Muestreamos k ~ Binomial(n_trials, p) de forma independiente
        const int n_trials = total_kfs - 1;
        std::binomial_distribution<int> binomial(n_trials, p);

        std::vector<int> result;
        result.reserve(n);

        for (int i = 0; i < n; i++) {
            int k = binomial(rng_);
            k = std::max(0, std::min(k, n_trials));
            // Espejamos el índice para sesgar hacia keyframes recientes
            k = n_trials - k;
            result.push_back(k);
        }

        std::sort(result.begin(), result.end());
        return result;
    }

    // ============================================================================
    // Método 2: Uniforme
    // ============================================================================

    std::vector<int> KeyframeSelector::sample_uniform(int n, int total_kfs,
                                                       const KeyframeSelectionConfig &config)
    {
        (void)config;
        return sample_without_replacement(n, total_kfs);
    }

    // ============================================================================
    // Método 3: Exponencial (bias hacia recientes)
    // ============================================================================

    std::vector<int> KeyframeSelector::sample_exponential(int n, int total_kfs,
                                                           const KeyframeSelectionConfig &config)
    {
        const float lambda = config.exponential_lambda;

        // Probabilidad exponencial: P(k) = exp(lambda * k / total_kfs)
        // Indexados de 0 a total_kfs-1, donde total_kfs-1 es el más reciente
        std::vector<float> probabilities(total_kfs);
        float sum_exp = 0.0f;

        for (int k = 0; k < total_kfs; k++) {
            float normalized_pos = static_cast<float>(k) / total_kfs;
            probabilities[k] = std::exp(lambda * normalized_pos);
            sum_exp += probabilities[k];
        }

        // Normalizar
        for (int k = 0; k < total_kfs; k++) {
            probabilities[k] /= sum_exp;
        }

        // Muestreo sin reemplazo usando Gumbel-Max
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::vector<std::pair<float, int>> scored_indices;

        for (int k = 0; k < total_kfs; k++) {
            float u = uniform(rng_);
            float gumbel = -std::log(-std::log(u + 1e-10f) + 1e-10f);
            float score = std::log(probabilities[k] + 1e-10f) + gumbel;
            scored_indices.push_back({score, k});
        }

        std::partial_sort(scored_indices.begin(), 
                         scored_indices.begin() + n,
                         scored_indices.end(),
                         [](const auto &a, const auto &b) { return a.first > b.first; });

        std::vector<int> result;
        for (int i = 0; i < n; i++) {
            result.push_back(scored_indices[i].second);
        }

        std::sort(result.begin(), result.end());
        return result;
    }

    // ============================================================================
    // Método 4: Sliding Window
    // ============================================================================

    std::vector<int> KeyframeSelector::sample_sliding_window(int n, int total_kfs,
                                                              const KeyframeSelectionConfig &config)
    {
        int window_size = config.sliding_window_window_size;
        if (window_size <= 0) {
            window_size = total_kfs / 2;
        }

        window_size = std::max(1, std::min(window_size, total_kfs));

        // Tomar los últimos 'window_size' keyframes
        // Luego muestrear uniformemente dentro de esa ventana
        int start_idx = std::max(0, total_kfs - window_size);
        int end_idx = total_kfs;

        std::vector<int> window_indices;
        for (int i = start_idx; i < end_idx; i++) {
            window_indices.push_back(i);
        }

        // Muestrear sin reemplazo de la ventana
        n = std::min(n, (int)window_indices.size());
        
        std::shuffle(window_indices.begin(), window_indices.end(), rng_);
        
        std::vector<int> result(window_indices.begin(), window_indices.begin() + n);
        std::sort(result.begin(), result.end());
        return result;
    }

    // ============================================================================
    // Utilidades: Distribuciones
    // ============================================================================

    float KeyframeSelector::sample_beta(float alpha, float beta)
    {
        // Método de composición: Beta(α,β) = Gamma(α) / (Gamma(α) + Gamma(β))
        float gamma_alpha = sample_gamma(alpha);
        float gamma_beta = sample_gamma(beta);
        float result = gamma_alpha / (gamma_alpha + gamma_beta);
        return std::max(0.0f, std::min(1.0f, result));
    }

    float KeyframeSelector::sample_gamma(float shape)
    {
        // Algoritmo de aceptación-rechazo para Gamma(shape)
        // Usa aproximación de Marsaglia y Tsang (2000)
        
        if (shape < 1.0f) {
            // Para shape < 1, usar: Gamma(shape) = Gamma(shape + 1) * U^(1/shape)
            std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
            float u = uniform(rng_);
            return sample_gamma(shape + 1.0f) * std::pow(u, 1.0f / shape);
        }

        float d = shape - 1.0f / 3.0f;
        float c = 1.0f / std::sqrt(9.0f * d);

        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

        while (true) {
            float z = normal(rng_);
            float v = 1.0f + c * z;
            
            if (v <= 0.0f) continue;
            
            v = v * v * v;
            float u = uniform(rng_);
            
            if (u < 1.0f - 0.0331f * z * z * z * z) {
                return d * v;
            }
            
            if (std::log(u) < 0.5f * z * z + d * (1.0f - v + std::log(v))) {
                return d * v;
            }
        }
    }

    std::vector<int> KeyframeSelector::sample_without_replacement(int n, int total)
    {
        std::vector<int> indices(total);
        std::iota(indices.begin(), indices.end(), 0);

        std::shuffle(indices.begin(), indices.end(), rng_);

        std::vector<int> result(indices.begin(), indices.begin() + n);
        std::sort(result.begin(), result.end());
        return result;
    }

} // namespace f_vigs_slam
