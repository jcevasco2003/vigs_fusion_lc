/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

 // Este archivo es identico al original, mas alla de algunos comentarios y prints eliminados para limpieza.

#include "f_vigs_slam/MarginalizationFactor.hpp"

#include <iostream>

namespace f_vigs_slam
{
    const int NUM_THREADS = 4;

    ResidualBlockInfo::ResidualBlockInfo(ceres::CostFunction *_cost_function,
                                         ceres::LossFunction *_loss_function,
                                         std::vector<double *> _parameter_blocks,
                                         std::vector<int> _drop_set)
        : cost_function(_cost_function), 
          loss_function(_loss_function), 
          parameter_blocks(_parameter_blocks), 
          drop_set(_drop_set), 
          raw_jacobians(0)
    {
        // Reservar memoria para residuales
        residuals.resize(cost_function->num_residuals());

        // Obtener tamaños de bloques y reservar memoria para jacobianos
        std::vector<int> block_sizes = cost_function->parameter_block_sizes();
        raw_jacobians = new double *[block_sizes.size()];

        jacobians.resize(block_sizes.size());

        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            // Cada jacobiano es num_residuals × block_size
            jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
            raw_jacobians[i] = jacobians[i].data();
        }
    }

    void ResidualBlockInfo::Evaluate()
    {
        // Evaluar función de costo: obtiene residuales y jacobianos
        cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

        if (loss_function)
        {
            double residual_scaling_, alpha_sq_norm_;

            double sq_norm, rho[3];

            // Norma cuadrada del residual
            sq_norm = residuals.squaredNorm();

            // Evaluar función de pérdida: rho[0] = ρ(s), rho[1] = ρ'(s), rho[2] = ρ''(s)
            loss_function->Evaluate(sq_norm, rho);


            // printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

           
            double sqrt_rho1_ = sqrt(rho[1]);

            // Calcular factores de escalado según método de Ceres
            if ((sq_norm == 0.0) || (rho[2] <= 0.0))
            {
                // Caso lineal o convexo: solo escalado por sqrt(ρ'(s))
                residual_scaling_ = sqrt_rho1_;
                alpha_sq_norm_ = 0.0;
            }
            else
            {
                // Caso no-convexo: corrección adicional
                const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
                const double alpha = 1.0 - sqrt(D);
                residual_scaling_ = sqrt_rho1_ / (1 - alpha);
                alpha_sq_norm_ = alpha / sq_norm;
            }

            // Escalar jacobianos: J_scaled = sqrt(ρ') * (J - α * r * r^T * J)
            for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
            {
                jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
            }

            // Escalar residuales: r_scaled = sqrt(ρ'/(1-α)) * r
            residuals *= residual_scaling_;
        }
    }

    MarginalizationInfo::~MarginalizationInfo()
    {
        // ROS_WARN("release marginlizationinfo");

        // Liberar memoria de datos de parámetros copiado
        for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
            delete it->second;

        // Liberar memoria de bloques residuales
        for (int i = 0; i < (int)factors.size(); i++)
        {

            delete[] factors[i]->raw_jacobians;

            delete factors[i]->cost_function;

            delete factors[i];
        }
    }

    void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
    {
        factors.emplace_back(residual_block_info);

        std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
        std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

        // Registrar tamaños de bloques de parámetros
        for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
        {
            double *addr = parameter_blocks[i];
            int size = parameter_block_sizes[i];
            parameter_block_size[reinterpret_cast<long>(addr)] = size;

            // 	std::cout << "parameter block size : " << size << std::endl;
        }

        // Marcar parámetros a marginalizar (drop_set) con índice inicial 0
        for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
        {
            double *addr = parameter_blocks[residual_block_info->drop_set[i]];
            parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
        }
    }

    void MarginalizationInfo::init()
    {
        // Asignar índices a parámetros a marginalizar [0, m)
        int pos = 0;
        for (auto &it : parameter_block_idx)
        {
            it.second = pos;
            pos += localSize(parameter_block_size[it.first]);
        }

        m = pos; // Dimensión de variables a marginalizar

        // Asignar índices a parámetros a mantener [m, m+n)
        for (const auto &it : parameter_block_size)
        {
            if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
            {
                parameter_block_idx[it.first] = pos;
                pos += localSize(it.second);
            }
        }

        n = pos - m; // Dimensión de variables a mantener


        // ROS_INFO("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

        // Validar que hay variables a marginalizar
        if (m == 0)
        {
            valid = false;
            printf("unstable tracking...\n");
            return;
        }

        //     std::cout << "m = " << m << std::endl
        // 	      << "n = " << n << std::endl
        // 	<< "pos = " << pos << std::endl;


        // Inicializar matrices A (información) y b (residual ponderado)
        A = Eigen::MatrixXd(pos, pos);
        b = Eigen::VectorXd(pos);
        A.setZero();
        b.setZero();

        // Reservar espacio para jacobianos y residuales linearizados
        linearized_jacobians = Eigen::MatrixXd::Zero(m, m);
        linearized_residuals = Eigen::VectorXd::Zero(m);

        // Reservar memoria para datos de parámetros (punto de linearización)
        for (auto it : factors)
        {
            std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
            for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
            {
                long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
                int size = block_sizes[i];
                auto it_data = parameter_block_data.find(addr);
                if (it_data == parameter_block_data.end() || it_data->second == 0)
                {
                    // 		std::cout << "alloc param data"<<  i<< std::endl;
                    double *data = new double[size];
                    parameter_block_data[addr] = data;
                }
            }
        }
    }

    void MarginalizationInfo::preMarginalize()
    {
        // Evaluar todos los factores (calcular residuales y jacobianos)
        for (auto it : factors)
        {
            it->Evaluate();
        }

        // Copiar valores actuales de parámetros como punto de linearización
        for (auto it : factors)
        {
            std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
            for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
            {
                // 	    std::cout << "copy param " << i << std::endl;

                long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
                int size = block_sizes[i];
                auto it_data = parameter_block_data.find(addr);
                if (it_data == parameter_block_data.end() || it_data->second == 0)
                {
                    // 		std::cout << "alloc and copy param "<<  i<< std::endl;

                    // Asignar y copiar si no existe
                    double *data = new double[size];
                    memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                    parameter_block_data[addr] = data;
                }
                else
                {
                    // 		std::cout << "copy param "<<  i<< std::endl;

                    // Copiar a memoria existente
                    memcpy(it_data->second, it->parameter_blocks[i], sizeof(double) * size);
                }
            }
        }
    }

    int MarginalizationInfo::localSize(int size) const
    {
        return size == 7 ? 6 : size;
    }

    int MarginalizationInfo::globalSize(int size) const
    {
        return size == 6 ? 7 : size;
    }

    /**
     * @brief Función ejecutada por cada thread para construir A y b
     * 
     * Cada thread acumula J^T*J y J^T*r de sus factores asignados.
     * 
     * @param threadsstruct Puntero a ThreadsStruct con factores y matrices
     */
    void *ThreadsConstructA(void *threadsstruct)
    {
        // std::cout << "ThreadsConstructA begin" << std::endl;

        ThreadsStruct *p = ((ThreadsStruct *)threadsstruct);

        // Procesar cada factor asignado a este thread
        for (auto it : p->sub_factors)
        {
            for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
            {
                int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
                int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
                if (size_i == 7)
                    size_i = 6; // Convertir a dimensión local

                // Jacobiano respecto al i-ésimo parámetro (primeras size_i columnas)
                Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);

                // Construir bloques de A = J^T*J (simétrico)
                for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
                {
                    int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                    int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                    if (size_j == 7)
                        size_j = 6;
                    Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                    if (i == j)
                        // Bloque diagonal: J_i^T * J_i
                        p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    else
                    {
                        // Bloque fuera-diagonal: J_i^T * J_j (copiar transpuesto por simetría)
                        p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                        p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                    }
                }
                // Construir b = J^T*r
                p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
            }
        }

        // std::cout << "ThreadsConstructA end" << std::endl;
        return threadsstruct;
    }

    void MarginalizationInfo::marginalize()
    {
        //     std::cout << "MarginalizationInfo::marginalize()" << std::endl;
        //     std::cout << "m+n = " << m+n << std::endl;

        A.setZero();
        b.setZero();

        // Construir A y b usando threads paralelos
        pthread_t tids[NUM_THREADS];
        ThreadsStruct threadsstruct[NUM_THREADS];

        // Distribuir factores entre threads (round-robin)
        int i = 0;
        for (auto it : factors)
        {
            threadsstruct[i].sub_factors.push_back(it);
            i++;
            i = i % NUM_THREADS;
        }

        // Crear threads y ejecutar
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threadsstruct[i].A = Eigen::MatrixXd::Zero(m + n, m + n);
            threadsstruct[i].b = Eigen::VectorXd::Zero(m + n);
            threadsstruct[i].parameter_block_size = parameter_block_size;
            threadsstruct[i].parameter_block_idx = parameter_block_idx;
            int ret = pthread_create(&tids[i], NULL, ThreadsConstructA, (void *)&(threadsstruct[i]));
            if (ret != 0)
            {
                printf("[MarginalizationInfo] Error creando thread %d\n", i);
            }
        }

        // Esperar threads y acumular resultados
        for (int i = NUM_THREADS - 1; i >= 0; i--)
        {
            pthread_join(tids[i], NULL);
            A += threadsstruct[i].A;
            b += threadsstruct[i].b;
        }

        // Particionar matriz A en bloques según variables a marginalizar (m) y mantener (n)
        // A = [A_mm  A_mr]
        //     [A_rm  A_rr]
        Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());

        // Descomposición eigen para invertir A_mm de forma robusta
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

        // Pseudo-inversión: lambda_i > eps -> 1/lambda_i, sino 0
        Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
        // printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

        // Extraer bloques de A y b
        Eigen::VectorXd bmm = b.segment(0, m);
        Eigen::MatrixXd Amr = A.block(0, m, m, n);
        Eigen::MatrixXd Arm = A.block(m, 0, n, m);
        Eigen::MatrixXd Arr = A.block(m, m, n, n);
        Eigen::VectorXd brr = b.segment(m, n);

        // Complemento de Schur: elimina variables marginalizadas implícitamente
        // A* = A_rr - A_rm * A_mm^{-1} * A_mr
        // b* = b_r - A_rm * A_mm^{-1} * b_m
        Eigen::MatrixXd A_star = Arr - Arm * Amm_inv * Amr;
        Eigen::MatrixXd b_star = brr - Arm * Amm_inv * bmm;

        // Descomponer A* = J^T * J para obtener jacobiano linearizado
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A_star);

        // Construir J y r desde eigendecomposición
        // A* = V * Λ * V^T = (sqrt(Λ) * V^T)^T * (sqrt(Λ) * V^T)
        // J = sqrt(Λ) * V^T
        Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
        Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

        // Jacobiano linearizado: J
        linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();

        // Residual linearizado: r = J^{-1} * b* = sqrt(Λ^{-1}) * V^T * b*
        linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b_star;
    }

    std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
    {
        std::vector<double *> keep_block_addr;
        keep_block_size.clear();
        keep_block_idx.clear();
        keep_block_data.clear();

        // Recopilar bloques a mantener (índice >= m)
        for (const auto &it : parameter_block_idx)
        {
            if (it.second >= m)
            {
                keep_block_size.push_back(parameter_block_size[it.first]);
                keep_block_idx.push_back(parameter_block_idx[it.first]);
                keep_block_data.push_back(parameter_block_data[it.first]);
                keep_block_addr.push_back(addr_shift[it.first]);
            }
        }
        sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

        return keep_block_addr;
    }
    MarginalizationFactor::MarginalizationFactor()
    {
    }

    MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *_marginalization_info)
    {
        // marginalization_info->init();
        init(_marginalization_info);
    };

    void MarginalizationFactor::init(MarginalizationInfo *_marginalization_info)
    {
        marginalization_info = _marginalization_info;

        int cnt = 0;
        // Configurar tamaños de bloques de parámetros del factor
        for (auto it : marginalization_info->keep_block_size)
        {
            mutable_parameter_block_sizes()->push_back(it);
            cnt += it;
        }

        // Configurar número de residuales (dimensión n)
        set_num_residuals(marginalization_info->n);
    }

    bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        int n = marginalization_info->n;
        int m = marginalization_info->m;

        // Calcular diferencia dx = x - x_0 en espacio tangente (local)
        Eigen::VectorXd dx(n);
        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            int size = marginalization_info->keep_block_size[i];
            int idx = marginalization_info->keep_block_idx[i] - m;

            Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
            Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);

            if (size != 7)
                // Caso lineal: diferencia directa
                dx.segment(idx, size) = x - x0;
            else
            {
                // Caso SE(3): posición lineal + quaternion relativo
                // Posición: p - p_0
                dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();

                // Rotación: 2 * (q_0^{-1} (x) q).vec() (parte vectorial del quaternion relativo)
                // q_0 = [w, x, y, z] = [x0(6), x0(3), x0(4), x0(5)]
                dx.segment<3>(idx + 3) = 2.0 * (Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
                if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
                {
                    dx.segment<3>(idx + 3) = -2.0 * (Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
                }
            }
        }

        // Calcular residual: r = r_0 + J * dx
        Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;

        //     std::cout << "residuals : " << Eigen::Map<Eigen::VectorXd>(residuals, n).transpose() << std::endl;

        // Calcular jacobianos si se solicitan
        if (jacobians)
        {

            for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
            {
                if (jacobians[i])
                {
                    int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                    int idx = marginalization_info->keep_block_idx[i] - m;

                    // Jacobiano es simplemente las columnas correspondientes de J linearizado
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                    jacobian.setZero();
                    jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
                }
            }
        }
        return true;
    }

}; // namespace f_vigs_slam