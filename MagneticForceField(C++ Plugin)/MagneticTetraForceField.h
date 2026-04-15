#pragma once

#include "MagneticPluginConfig.h"

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>

#include <Eigen/Dense>
#include <string>
#include <Eigen/StdVector> // 必须包含此头文件以支持 vector<Matrix> This header file must be included to support vector <Matrix>
#include <vector>

namespace sofa::core::objectmodel
{
class Event;
}
namespace sofa::linearalgebra
{
class BaseMatrix;
}

namespace magneticplugin
{

class SOFA_MAGNETICPLUGIN_API MagneticTetraForceField
    : public sofa::core::behavior::ForceField<sofa::defaulttype::Vec3Types>
{
public:
    SOFA_CLASS(MagneticTetraForceField, sofa::core::behavior::ForceField<sofa::defaulttype::Vec3Types>);

    // 这一行宏是必须的，为了保证类成员中的 Eigen 固定尺寸类型正确对齐 This macro line is necessary to ensure that the Eigen fixed-size type in the class members is correctly aligned.
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Inherited = sofa::core::behavior::ForceField<sofa::defaulttype::Vec3Types>;
    using Real = Inherited::Real;
    using SReal = ::SReal;
    using DataVecCoord = Inherited::DataVecCoord;
    using DataVecDeriv = Inherited::DataVecDeriv;
    
    using SofaVec3 = sofa::type::Vec<3, Real>;
    using Vec4i = sofa::type::Vec4i;

    // Eigen Type
    using Mat3 = Eigen::Matrix<Real, 3, 3>;
    using Vec3 = Eigen::Matrix<Real, 3, 1>;

    MagneticTetraForceField();

    void init() override;

    void addForce(const sofa::core::MechanicalParams* mparams,
                  DataVecDeriv& f,
                  const DataVecCoord& x,
                  const DataVecDeriv& v) override;

    void addDForce(const sofa::core::MechanicalParams* mparams,
                   DataVecDeriv& df,
                   const DataVecDeriv& dx) override;

    void addKToMatrix(sofa::linearalgebra::BaseMatrix* matrix,
                      SReal kFact,
                      unsigned int& offset) override;

    SReal getPotentialEnergy(const sofa::core::MechanicalParams* mparams,
                             const DataVecCoord& x) const override;

    void handleEvent(sofa::core::objectmodel::Event* event) override;

protected:
    sofa::core::objectmodel::Data<SofaVec3> d_B;
    sofa::core::objectmodel::Data<SofaVec3> d_M0;
    sofa::core::objectmodel::Data<Real> d_scaleFactor;
    sofa::core::objectmodel::Data<Real> d_profileWindow;
    sofa::core::objectmodel::Data<std::string> d_profileOutput;
    sofa::core::objectmodel::Data<int> d_profileSampleStride;

    sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer* m_topology{nullptr};

    std::vector<Vec4i> m_tetIndices;
    
    // 【修正】使用 aligned_allocator 防止崩溃 [Correction] Use aligned_allocator to prevent crashes
    std::vector<Mat3, Eigen::aligned_allocator<Mat3>> m_DmInv;
    
    std::vector<bool> m_validTet;

private:
    // 内部帮助函数：使用迭代法计算旋转，避免 SVD 内存分配 Internal helper function: Uses an iterative method to calculate rotations, avoiding SVD memory allocation.
    static Mat3 computePolarRotation(const Mat3& F);

    bool m_profileDone{false};
    bool m_profileWritePending{false};
    size_t m_profileCount{0};
    size_t m_profileCapacity{0};
    int m_profileSampleStride{1};
    Real m_profileWindowValue{Real(0.0)};
    Real m_profileLastTime{Real(-1.0)};
    Real m_profileWallStartSeconds{Real(0.0)};
    bool m_profileWallStartSet{false};
    std::string m_profileOutputPath;
    std::vector<Real> m_profileSimTimes;
    std::vector<Real> m_profileWallTimes;
    std::vector<Real> m_profileDurationsMs;
    std::vector<Real> m_profileRotationMs;
    std::vector<Real> m_profileFaceMs;
    std::vector<size_t> m_profileSampleCounts;
};

} // namespace magneticplugin
