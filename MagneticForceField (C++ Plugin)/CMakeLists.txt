cmake_minimum_required(VERSION 3.22)
project(MagneticPlugin VERSION 1.0)

# Build the SOFA magnetic-force plugin as a shared library.
# 将 SOFA 磁力场插件构建为动态链接库。

find_package(Sofa.Framework REQUIRED)
find_package(Sofa.Helper REQUIRED)
find_package(Sofa.Core REQUIRED)
find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
find_package(Eigen3 REQUIRED)

set(HEADER_FILES
    MagneticPluginConfig.h
    initMagneticPlugin.h
    MagneticTetraForceField.h
)

set(SOURCE_FILES
    MagneticTetraForceField.cpp
    initMagneticPlugin.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

# Expose local headers and Eigen includes to downstream targets.
# 向下游目标公开本地头文件目录和 Eigen 头文件。

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    Sofa.Helper
    Sofa.Core
    Sofa.Framework
    Sofa.Component.Topology.Container.Dynamic
    Eigen3::Eigen
)

target_compile_definitions(${PROJECT_NAME} PRIVATE SOFA_BUILD_MAGNETICPLUGIN)

install(TARGETS ${PROJECT_NAME}
    # Install the plugin into the usual runtime and library folders.
    # 将插件安装到常规的运行时与库目录。
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
