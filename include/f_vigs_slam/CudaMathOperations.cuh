#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace f_vigs_slam
{
/**
 * @brief Suma componente a componente de dos vectores float3.
 */
__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Resta componente a componente de dos vectores float3.
 */
__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * @brief Multiplica un vector float3 por un escalar.
 */
__host__ __device__ inline float3 operator*(const float3 &a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

/**
 * @brief Multiplica un escalar por un vector float3.
 */
__host__ __device__ inline float3 operator*(float s, const float3 &a)
{
    return a * s;
}

/**
 * @brief Acumula suma componente a componente sobre el primer operando.
 * @param a [INOUT] Acumulador y resultado.
 * @param b [IN] Vector a acumular.
 * @return Referencia al acumulador actualizado.
 */
__host__ __device__ inline float3 &operator+=(float3 &a, const float3 &b)
{
    a = a + b;
    return a;
}

__device__ __forceinline__ float dot3(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief Calcula el producto cruz.
 */
__host__ __device__ inline float3 cross(const float3 &a, const float3 &b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

/**
 * @brief Máximo componente a componente entre dos vectores float3.
 * @param a [IN] Primer vector.
 * @param b [IN] Segundo vector.
 * @return Vector con máximos por componente.
 */
__host__ __device__ inline float3 vec3Max(const float3 &a, const float3 &b)
{
    return make_float3(
        a.x > b.x ? a.x : b.x,
        a.y > b.y ? a.y : b.y,
        a.z > b.z ? a.z : b.z);
}

/**
 * @brief Mínimo componente a componente entre dos vectores float3.
 */
__host__ __device__ inline float3 vec3Min(const float3 &a, const float3 &b)
{
    return make_float3(
        a.x < b.x ? a.x : b.x,
        a.y < b.y ? a.y : b.y,
        a.z < b.z ? a.z : b.z);
}

/**
 * @brief Normaliza un vector float3 con protección numérica.
 */
__host__ __device__ inline float3 normalizeVec3(const float3 &v)
{
    float n = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (n > 1e-8f)
    {
        float inv = 1.0f / n;
        return make_float3(v.x * inv, v.y * inv, v.z * inv);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

/**
 * @brief Rota un vector por un cuaternión unitario q.
 * @param q [IN] Cuaternión en formato (x, y, z, w).
 * @param v [IN] Vector a rotar.
 * @return Vector rotado q * v * q^{-1}.
 */
__host__ __device__ inline float3 rotateByQuaternion(const float4 &q, const float3 &v)
{
    float3 qv = make_float3(q.x, q.y, q.z);
    float3 t = 2.0f * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

/**
 * @brief Multiplica dos cuaterniones.
 * @param a [IN] Primer cuaternión en formato (x, y, z, w).
 * @param b [IN] Segundo cuaternión en formato (x, y, z, w).
 * @return Cuaternión producto a * b.
 */
__host__ __device__ inline float4 quatMultiply(const float4 &a, const float4 &b)
{
    float ax = a.x, ay = a.y, az = a.z, aw = a.w;
    float bx = b.x, by = b.y, bz = b.z, bw = b.w;
    float4 out;
    out.x = aw * bx + ax * bw + ay * bz - az * by;
    out.y = aw * by - ax * bz + ay * bw + az * bx;
    out.z = aw * bz + ax * by - ay * bx + az * bw;
    out.w = aw * bw - ax * bx - ay * by - az * bz;
    return out;
}

/**
 * @brief Obtiene el cuaternión que alinea un vector origen con un vector destino.
 * @param from [IN] Vector origen.
 * @param to [IN] Vector destino.
 * @return Cuaternión unitario que rota from hacia to.
 */
__host__ __device__ inline float4 quatFromTwoVectors(const float3 &from, const float3 &to)
{
    float3 u = normalizeVec3(from);
    float3 v = normalizeVec3(to);
    float dot = u.x * v.x + u.y * v.y + u.z * v.z;

    if (dot < -0.999999f)
    {
        float3 axis = (fabsf(u.x) < 0.1f) ? make_float3(1.0f, 0.0f, 0.0f)
                                          : make_float3(0.0f, 1.0f, 0.0f);
        float3 ortho = normalizeVec3(cross(u, axis));
        return make_float4(ortho.x, ortho.y, ortho.z, 0.0f);
    }

    float3 c = cross(u, v);
    float4 q = make_float4(c.x, c.y, c.z, 1.0f + dot);
    float n = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n > 1e-8f)
    {
        float inv = 1.0f / n;
        q.x *= inv;
        q.y *= inv;
        q.z *= inv;
        q.w *= inv;
    }
    else
    {
        q = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return q;
}

/**
 * @brief Invierte un cuaternión unitario mediante conjugado.
 * @param q [IN] Cuaternión en formato (x, y, z, w).
 * @return Cuaternión inverso.
 */
__host__ __device__ inline float4 quaternionInverse(const float4 &q)
{
    return make_float4(-q.x, -q.y, -q.z, q.w);
}

/**
 * @brief Rota un vector por el inverso de un cuaternión.
 * @param q [IN] Cuaternión en formato (x, y, z, w).
 * @param v [IN] Vector a rotar.
 * @return Vector rotado q^{-1} * v * q.
 */
__host__ __device__ inline float3 rotateByQuaternionInverse(const float4 &q, const float3 &v)
{
    float3 qv = make_float3(q.x, q.y, q.z);
    float w = q.w;

    float3 t = make_float3(
        qv.y * v.z - qv.z * v.y,
        qv.z * v.x - qv.x * v.z,
        qv.x * v.y - qv.y * v.x);

    float3 crossTerm = make_float3(
        qv.y * t.z - qv.z * t.y,
        qv.z * t.x - qv.x * t.z,
        qv.x * t.y - qv.y * t.x);

    float3 result;
    result.x = v.x + 2.f * (-w * t.x + crossTerm.x);
    result.y = v.y + 2.f * (-w * t.y + crossTerm.y);
    result.z = v.z + 2.f * (-w * t.z + crossTerm.z);
    return result;
}

/**
 * @brief Convierte un cuaternión a matriz de rotación 3x3.
 * @param q [IN] Cuaternión en formato (x, y, z, w).
 * @param R [OUT] Matriz de rotación de salida.
 */
__host__ __device__ inline void quaternionToMatrix(const float4 &q, float R[3][3])
{
    float w = q.w, x = q.x, y = q.y, z = q.z;

    R[0][0] = 1.0f - 2.0f * (y * y + z * z);
    R[0][1] = 2.0f * (x * y - w * z);
    R[0][2] = 2.0f * (x * z + w * y);

    R[1][0] = 2.0f * (x * y + w * z);
    R[1][1] = 1.0f - 2.0f * (x * x + z * z);
    R[1][2] = 2.0f * (y * z - w * x);

    R[2][0] = 2.0f * (x * z - w * y);
    R[2][1] = 2.0f * (y * z + w * x);
    R[2][2] = 1.0f - 2.0f * (x * x + y * y);
}

/**
 * @brief Multiplica dos matrices 3x3.
 * @param A [IN] Matriz izquierda.
 * @param B [IN] Matriz derecha.
 * @param R [OUT] Resultado A * B.
 */
__host__ __device__ inline void mult3x3(const float A[3][3], const float B[3][3], float R[3][3])
{
    float temp[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp[i][j] = 0.0f;
            for (int k = 0; k < 3; k++) {
                temp[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R[i][j] = temp[i][j];
        }
    }
}

/**
 * @brief Multiplica matriz 3x3 por vector 3D.
 * @param R [IN] Matriz de entrada.
 * @param v [IN] Vector de entrada.
 * @return Resultado R * v.
 */
__host__ __device__ inline float3 matrixVectorMul(const float R[3][3], const float3 &v)
{
    return make_float3(
        R[0][0] * v.x + R[0][1] * v.y + R[0][2] * v.z,
        R[1][0] * v.x + R[1][1] * v.y + R[1][2] * v.z,
        R[2][0] * v.x + R[2][1] * v.y + R[2][2] * v.z);
}

/**
 * @brief Invierte una matriz 3x3 por adjunta y determinante.
 * @param A [IN] Matriz de entrada.
 * @param invA [OUT] Matriz inversa; si es singular se llena con ceros.
 */
__host__ __device__ inline void invert3x3(const float A[3][3], float invA[3][3])
{
    float det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
              - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
              + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);

    if (fabsf(det) < 1e-6f) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                invA[i][j] = 0.0f;
        return;
    }

    float invDet = 1.0f / det;

    invA[0][0] =  (A[1][1]*A[2][2] - A[1][2]*A[2][1]) * invDet;
    invA[0][1] = -(A[0][1]*A[2][2] - A[0][2]*A[2][1]) * invDet;
    invA[0][2] =  (A[0][1]*A[1][2] - A[0][2]*A[1][1]) * invDet;

    invA[1][0] = -(A[1][0]*A[2][2] - A[1][2]*A[2][0]) * invDet;
    invA[1][1] =  (A[0][0]*A[2][2] - A[0][2]*A[2][0]) * invDet;
    invA[1][2] = -(A[0][0]*A[1][2] - A[0][2]*A[1][0]) * invDet;

    invA[2][0] =  (A[1][0]*A[2][1] - A[1][1]*A[2][0]) * invDet;
    invA[2][1] = -(A[0][0]*A[2][1] - A[0][1]*A[2][0]) * invDet;
    invA[2][2] =  (A[0][0]*A[1][1] - A[0][1]*A[1][0]) * invDet;
}

/**
 * @brief Limita un índice entero al rango [min_value, max_value].
 * @param value [IN] Índice original.
 * @param min_value [IN] Límite inferior.
 * @param max_value [IN] Límite superior.
 * @return Índice acotado al rango.
 */
__host__ __device__ inline int clampIndex(int value, int min_value, int max_value)
{
    return value < min_value ? min_value : (value > max_value ? max_value : value);
}

template <typename T>
struct Texture2DView
{
    cudaTextureObject_t texture;

    __device__ __forceinline__ T load(int x, int y) const
    {
        return tex2D<T>(texture, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
    }
};

template <typename T>
__host__ __device__ __forceinline__ T getImageData(
    const T *image,
    size_t image_step,
    int x,
    int y)
{
    const char *row_ptr = reinterpret_cast<const char *>(image) + static_cast<size_t>(y) * image_step;
    const T *row = reinterpret_cast<const T *>(row_ptr);
    return row[x];
}

template <typename T>
__device__ __forceinline__ T getImageData(
    cudaTextureObject_t texture,
    int x,
    int y)
{
    return tex2D<T>(texture, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
}

template <typename T>
__host__ __device__ __forceinline__ void setImageData(
    T *image,
    size_t image_step,
    int x,
    int y,
    const T &value)
{
    char *row_ptr = reinterpret_cast<char *>(image) + static_cast<size_t>(y) * image_step;
    T *row = reinterpret_cast<T *>(row_ptr);
    row[x] = value;
}

struct ToFloat3
{
    __host__ __device__ float3 operator()(const float4 &p) const
    {
        return make_float3(p.x, p.y, p.z);
    }
};

template <typename T>
__device__ __forceinline__ void deviceSwap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return m;
}

__device__ __forceinline__ void skewSymmetric(const float3& v, float out[9])
{
    out[0] = 0.0f;   out[1] = -v.z;  out[2] =  v.y;
    out[3] = v.z;    out[4] = 0.0f;  out[5] = -v.x;
    out[6] = -v.y;   out[7] = v.x;   out[8] = 0.0f;
}

__device__ __forceinline__
uint32_t atomicAggInc(uint32_t* ctr)
{
    // Máscara de threads activos en el warp
    unsigned int active = __activemask();

    // Lane real dentro del warp (válido para bloques 1D/2D/3D)
    const unsigned int linear_tid =
        static_cast<unsigned int>(threadIdx.x) +
        static_cast<unsigned int>(blockDim.x) *
            (static_cast<unsigned int>(threadIdx.y) +
             static_cast<unsigned int>(blockDim.y) * static_cast<unsigned int>(threadIdx.z));
    const unsigned int lane = (linear_tid & 31u);

    // Elegimos líder (primer thread activo)
    int leader = __ffs(active) - 1;

    // Cantidad de threads activos en el warp
    int change = __popc(active);

    // Rank del thread dentro del warp activo
    int rank = __popc(active & ((1u << lane) - 1u));

    uint32_t base = 0;

    // Solo el líder hace el atomic
    if (rank == 0)
    {
        base = atomicAdd(ctr, change);
    }

    // Broadcast del resultado a todo el warp
    base = __shfl_sync(active, base, leader);

    return base + rank;
}

} // namespace f_vigs_slam
