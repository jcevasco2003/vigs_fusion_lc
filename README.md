### Fue hecho con:

### ROS2 Rolling

### Cuda 12.3 con arquitectura 6.1

### OpenCV 4.8.0 bajo la forma de instalación:

Compilado con g10/g++10

cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.8.0
cd ../opencv_contrib
git checkout 4.8.0
cd ~/opencv
mkdir build && cd build

CC=gcc-10 CXX=g++-10 cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D WITH_TBB=ON \
-D WITH_GTK=ON \
-D WITH_QT=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D CUDA_ARCH_BIN=6.1 \
-D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3.10/dist-packages \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_cudaarithm=ON \
-D BUILD_opencv_cudaimgproc=ON \
-D BUILD_opencv_cudawarping=ON \
-D BUILD_opencv_cudafilters=ON \
-D BUILD_opencv_java=OFF \
..

luego aplicando la corrección https://github.com/opencv/opencv/issues/23893

### Suitesparse

Con g12/g++12

cd ~
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse
git checkout v5.10.1
make -j$(nproc)
sudo make install
sudo ldconfig

### Eigen

### glog

### Ceres Solver

cd ~
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
git checkout 2.0.0
mkdir build && cd build
cmake .. \
-DBUILD_TESTING=OFF \
-DBUILD_EXAMPLES=OFF \
-DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig

#### Guia de uso

Actualmente, para usarlo con cualquier Dataset de VIGS-Fusion, disponibles en https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/CI0K9G

- Como el dataset guarda la imagen rgb comprimida, para descomprimir la imagen usar un nodo que publique a /camera/color/image_raw

- Compilar con colcon build
- Activar con:
- ros2 launch f_vigs_slam gs_slam_node.py


- Qué es cada parámetro:
- `use_sim_time`: usa reloj simulado de ROS (`/clock`). Para bags/sensores reales, normalmente `false`.
- `imu_topic`: topico de IMU cruda/filtrada que se integra en el pipeline.
- `imu_preint_topic`: topico de IMU para suscripcion diagnostica de preintegracion.
- `depth_topic`: topico de imagen de profundidad.
- `color_topic`: topico de imagen RGB.
- `camera_info_topic`: topico de intrinsecas y frame de camara.
- `downsample_factor`: `1` o entero par (`2, 4, 6, ...`). Hace downsample de RGB/depth y ajusta intrinsecas.
- `covisibility_threshold`: umbral de covisibilidad para decidir/filtrar seleccion de keyframes.
- `initialization_strategy`: estrategia de muestreo para inicialización (ver sección "Estrategias de muestreo regional").
- `densification_strategy`: estrategia de muestreo para densificación (ver sección "Estrategias de muestreo regional").
- `imu_reprop_ba_thresh`: umbral de cambio en bias de acelerometro para forzar repropagacion IMU en el factor.
- `imu_reprop_bg_thresh`: umbral de cambio en bias de giroscopio para repropagacion IMU.
- `imu_init_samples`: cantidad de muestras IMU usadas para inicializar gravedad/bias.
- `imu_dt_warn_max_s`: `dt` maximo para marcar advertencias de muestreo IMU anomalo.
- `imu_acc_norm_min`: norma minima esperada de aceleracion para diagnostico.
- `imu_acc_norm_max`: norma maxima esperada de aceleracion para diagnostico.
- `imu_gyro_norm_max`: norma maxima esperada de velocidad angular para diagnostico.
- `diag_state_jump_pos_thresh_m`: umbral de salto de posicion para marcar `abrupt state jump`.
- `diag_state_jump_rot_thresh_deg`: umbral de salto angular (grados) para `abrupt state jump`.
- `diag_proc_time_domain_abs_limit_s`: limite para invalidar `lag_proc` cuando reloj de nodo y stamps no coinciden.
- `publish_pointcloud`: habilita/deshabilita publicacion de nube de gaussianas (`gaussian_pointcloud`).
- `publish_reconstructed_images`: habilita/deshabilita publicacion de imagen reconstruida (`reconstructed_image`).
- `evaluate_metrics`: habilita salida de metricas de rendimiento/evaluacion.
- `perf_baseline_tag`: etiqueta textual para identificar la corrida en CSV/logs de performance.
- `perf_csv_every_n_frames`: frecuencia de volcado a CSV cada N frames procesados.


