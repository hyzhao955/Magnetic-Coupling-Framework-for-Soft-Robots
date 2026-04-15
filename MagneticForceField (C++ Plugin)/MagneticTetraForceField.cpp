#include "MagneticTetraForceField.h"

// Implementation of the tetrahedral magnetic force field used in the plugin.
// 插件中四面体磁力场的实现文件。

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/VecId.h>
#include <sofa/helper/logging/Messaging.h>

#include <chrono>
#include <cmath>
#include <fstream>

namespace magneticplugin
{

MagneticTetraForceField::MagneticTetraForceField()
    : d_B(initData(&d_B, SofaVec3(0.0, 0.0, 0.0), "B", "Magnetic field"))
    , d_M0(initData(&d_M0, SofaVec3(0.0, 0.0, 0.0), "M0", "Rest magnetization"))
    , d_scaleFactor(initData(&d_scaleFactor, Real(1.0), "scaleFactor", "Force scaling"))
    , d_profileWindow(initData(&d_profileWindow, Real(1.0), "profileWindow", "Profiling window (s) for addForce timing"))
    , d_profileOutput(initData(&d_profileOutput, std::string("magnetic_addforce_profile.csv"), "profileOutput",
        "CSV output path for addForce timing"))
    , d_profileSampleStride(initData(&d_profileSampleStride, 10, "profileSampleStride",
        "Sample stride for per-tet profiling (>=1)"))
{
}

void MagneticTetraForceField::init()
{
    // Cache topology and reference-state data needed later in addForce().
    // 缓存后续 addForce() 所需的拓扑和参考状态数据。
    Inherited::init();
    this->f_listening.setValue(true);

    m_topology = this->getContext()->get<sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>();
    if (!m_topology)
    {
        msg_error() << "MagneticTetraForceField requires a TetrahedronSetTopologyContainer";
        return;
    }

    auto* mstate = this->getMState();
    if (!mstate)
    {
        msg_error() << "MagneticTetraForceField requires a MechanicalState";
        return;
    }

    // 【修正】安全地获取初始位置，避免悬垂引用
    // 我们将数据拷贝到本地 vector 中，这样生命周期由我们控制
    const auto* restData = mstate->read(sofa::core::vec_id::read_access::restPosition);
    const auto* posData = mstate->read(sofa::core::vec_id::read_access::position);
    if (!restData || !posData)
    {
        msg_error() << "MagneticTetraForceField failed to access positions data";
        return;
    }

    const auto& rest = restData->getValue();
    const auto& pos = posData->getValue();
    const auto& positionsRef = (!rest.empty()) ? rest : pos;

    const auto& tets = m_topology->getTetrahedra();
    m_tetIndices.clear();
    m_tetIndices.reserve(tets.size());
    for (const auto& tet : tets)
    {
        m_tetIndices.emplace_back(tet[0], tet[1], tet[2], tet[3]);
    }
    m_DmInv.resize(m_tetIndices.size());
    m_validTet.assign(m_tetIndices.size(), false);

    for (size_t i = 0; i < m_tetIndices.size(); ++i)
    {
        const Vec4i& tet = m_tetIndices[i];
        
        // 这里的 init 检查是静态的，运行时 addForce 还需要检查
        if (tet[0] >= positionsRef.size() || tet[1] >= positionsRef.size() || 
            tet[2] >= positionsRef.size() || tet[3] >= positionsRef.size())
        {
            msg_warning() << "Tetrahedron " << i << " references out of bound nodes. Ignored.";
            continue;
        }

        const SofaVec3& XaRef = positionsRef[tet[0]];
        const SofaVec3& XbRef = positionsRef[tet[1]];
        const SofaVec3& XcRef = positionsRef[tet[2]];
        const SofaVec3& XdRef = positionsRef[tet[3]];
        Vec3 Xa(XaRef[0], XaRef[1], XaRef[2]);
        Vec3 Xb(XbRef[0], XbRef[1], XbRef[2]);
        Vec3 Xc(XcRef[0], XcRef[1], XcRef[2]);
        Vec3 Xd(XdRef[0], XdRef[1], XdRef[2]);

        Mat3 Dm;
        Dm.col(0) = Xb - Xa;
        Dm.col(1) = Xc - Xa;
        Dm.col(2) = Xd - Xa;

        if (std::abs(Dm.determinant()) < 1e-12)
        {
            msg_warning() << "Tetrahedron " << i << " has zero volume (degenerate). Ignored.";
            m_validTet[i] = false;
        }
        else
        {
            m_DmInv[i] = Dm.inverse();
            m_validTet[i] = true;
        }
    }

    m_profileWindowValue = d_profileWindow.getValue();
    m_profileOutputPath = d_profileOutput.getValue();
    m_profileSampleStride = d_profileSampleStride.getValue();
    if (m_profileSampleStride < 1)
    {
        m_profileSampleStride = 1;
    }
    m_profileCount = 0;
    m_profileDone = false;
    m_profileWritePending = false;
    m_profileLastTime = Real(-1.0);
    m_profileWallStartSeconds = Real(0.0);
    m_profileWallStartSet = false;

    if (m_profileWindowValue > Real(0.0))
    {
        const auto* ctx = this->getContext();
        Real dt = ctx ? static_cast<Real>(ctx->getDt()) : Real(0.0);
        if (dt <= Real(0.0))
        {
            dt = Real(0.03);
        }
        m_profileCapacity = static_cast<size_t>(std::ceil(m_profileWindowValue / dt)) + 2;
        m_profileSimTimes.assign(m_profileCapacity, Real(0.0));
        m_profileWallTimes.assign(m_profileCapacity, Real(0.0));
        m_profileDurationsMs.assign(m_profileCapacity, Real(0.0));
        m_profileRotationMs.assign(m_profileCapacity, Real(0.0));
        m_profileFaceMs.assign(m_profileCapacity, Real(0.0));
        m_profileSampleCounts.assign(m_profileCapacity, 0);
    }
    else
    {
        m_profileCapacity = 0;
    }
}

MagneticTetraForceField::Mat3 MagneticTetraForceField::computePolarRotation(const Mat3& F)
{
    // Iteratively extract the polar rotation from the deformation gradient.
    // 通过迭代方式从形变梯度中提取极分解旋转矩阵。
    Mat3 R = F;
    const int maxIter = 10;
    const Real tol = 1e-6;
    const Real detEps = 1e-12;

    for (int i = 0; i < maxIter; ++i)
    {
        if (!R.allFinite())
        {
            break;
        }

        const Real det = R.determinant();
        if (std::abs(det) < detEps)
        {
            break;
        }
        Mat3 R_inv_T = R.inverse().transpose();
        Mat3 R_next = 0.5 * (R + R_inv_T);
        
        // 简单的收敛检查 (L1 norm)
        if ((R_next - R).cwiseAbs().sum() < tol) 
        {
            R = R_next;
            break;
        }
        R = R_next;
    }

    // 确保行列式为正（处理反射）
    if (R.determinant() < 0)
    {
        // 这是一个比较粗暴的修正，但在物理迭代中通常足够
        // 更严格的做法需要重新分解，但此处我们简单取反
        R.col(2) *= Real(-1.0); 
    }
    return R;
}

void MagneticTetraForceField::addForce(const sofa::core::MechanicalParams* mparams,
                                       DataVecDeriv& f,
                                       const DataVecCoord& x,
                                       const DataVecDeriv& v)
{
    // Assemble equivalent nodal forces from the magnetic body torque in each tetrahedron.
    // 将每个四面体中的磁体力矩密度装配成等效节点力。
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(v);

    const auto* ctx = this->getContext();
    const Real simTime = ctx ? static_cast<Real>(ctx->getTime()) : Real(0.0);
    Real dtSim = (mparams != nullptr) ? static_cast<Real>(mparams->dt()) : Real(0.0);
    if (dtSim <= Real(0.0))
    {
        dtSim = ctx ? static_cast<Real>(ctx->getDt()) : Real(0.0);
    }
    if (dtSim <= Real(0.0))
    {
        dtSim = Real(0.03);
    }
    const Real profileEps = Real(1e-12);
    const bool doProfile = (!m_profileDone) &&
        (m_profileWindowValue > Real(0.0)) &&
        (simTime <= m_profileWindowValue + profileEps) &&
        (simTime > m_profileLastTime + profileEps) &&
        (m_profileCount < m_profileCapacity);
    std::chrono::high_resolution_clock::time_point profileStart;
    Real rotationAccum = Real(0.0);
    Real faceAccum = Real(0.0);
    size_t sampleCount = 0;
    if (doProfile)
    {
        profileStart = std::chrono::high_resolution_clock::now();
    }

    const auto& xVec = x.getValue();
    size_t nNodes = xVec.size(); // 获取当前节点数
    auto& fVec = *f.beginEdit();

    const SofaVec3& BRef = d_B.getValue();
    const SofaVec3& M0Ref = d_M0.getValue();
    Vec3 B(BRef[0], BRef[1], BRef[2]);
    Vec3 M0(M0Ref[0], M0Ref[1], M0Ref[2]);
    const Real scale = d_scaleFactor.getValue();

    if (scale == Real(0.0) || m_tetIndices.empty())
    {
        f.endEdit();
        return;
    }

    const Real scaleFactor = scale / Real(6.0);

    for (size_t i = 0; i < m_tetIndices.size(); ++i)
    {
        if (!m_validTet[i]) continue;

        const Vec4i& tet = m_tetIndices[i];
        
        // 【修正】运行时边界检查 (防止拓扑动态变化导致越界)
        if (tet[0] >= nNodes || tet[1] >= nNodes || tet[2] >= nNodes || tet[3] >= nNodes)
            continue;

        unsigned int idx[4] = {(unsigned int)tet[0], (unsigned int)tet[1], (unsigned int)tet[2], (unsigned int)tet[3]};

        const bool profileTet = doProfile &&
            (m_profileSampleStride <= 1 || (i % static_cast<size_t>(m_profileSampleStride) == 0));
        std::chrono::high_resolution_clock::time_point segmentStart;
        if (profileTet)
        {
            segmentStart = std::chrono::high_resolution_clock::now();
        }

        const Coord& xaRef = xVec[idx[0]];
        const Coord& xbRef = xVec[idx[1]];
        const Coord& xcRef = xVec[idx[2]];
        const Coord& xdRef = xVec[idx[3]];
        Vec3 xa(xaRef[0], xaRef[1], xaRef[2]);
        Vec3 xb(xbRef[0], xbRef[1], xbRef[2]);
        Vec3 xc(xcRef[0], xcRef[1], xcRef[2]);
        Vec3 xd(xdRef[0], xdRef[1], xdRef[2]);

        Mat3 Ds;
        Ds.col(0) = xb - xa;
        Ds.col(1) = xc - xa;
        Ds.col(2) = xd - xa;

        Mat3 F = Ds * m_DmInv[i];

        Mat3 R = computePolarRotation(F);

        Vec3 M = R * M0;
        Vec3 tau = M.cross(B);

        if (profileTet)
        {
            const auto segmentMid = std::chrono::high_resolution_clock::now();
            rotationAccum +=
                std::chrono::duration_cast<std::chrono::duration<Real, std::milli>>(segmentMid - segmentStart).count();
            segmentStart = segmentMid;
        }

        // 【修正】删除死代码，变量命名更清晰
        auto addFaceLoad = [&](const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& opp, 
                               unsigned int i0, unsigned int i1, unsigned int i2)
        {
            Vec3 n_til = (p1 - p0).cross(p2 - p0);
            Real area2 = n_til.norm(); 
            
            if (area2 < 1e-15) return;

            Vec3 n = n_til / area2;
            Real area = 0.5 * area2;

            Vec3 center = (p0 + p1 + p2) / 3.0;
            if (n.dot(opp - center) > 0) n = -n;

            // 使用修正后的干净逻辑
            // Vec3 force = n.cross(tau) * (area * scaleFactor);
            Vec3 force = tau.cross(n) * (area * scaleFactor); //修正了由于公式符号错误造成的force的方向颠倒

            fVec[i0][0] += force[0]; fVec[i0][1] += force[1]; fVec[i0][2] += force[2];
            fVec[i1][0] += force[0]; fVec[i1][1] += force[1]; fVec[i1][2] += force[2];
            fVec[i2][0] += force[0]; fVec[i2][1] += force[1]; fVec[i2][2] += force[2];
        };

        addFaceLoad(xb, xc, xd, xa, idx[1], idx[2], idx[3]);
        addFaceLoad(xa, xd, xc, xb, idx[0], idx[3], idx[2]);
        addFaceLoad(xa, xb, xd, xc, idx[0], idx[1], idx[3]);
        addFaceLoad(xa, xc, xb, xd, idx[0], idx[2], idx[1]);

        if (profileTet)
        {
            const auto segmentEnd = std::chrono::high_resolution_clock::now();
            faceAccum +=
                std::chrono::duration_cast<std::chrono::duration<Real, std::milli>>(segmentEnd - segmentStart).count();
            ++sampleCount;
        }
    }

    if (doProfile)
    {
        const auto profileEnd = std::chrono::high_resolution_clock::now();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<Real, std::milli>>(profileEnd - profileStart).count();

        m_profileSimTimes[m_profileCount] = simTime;
        m_profileDurationsMs[m_profileCount] = elapsed;
        const Real wallNow = static_cast<Real>(
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        if (!m_profileWallStartSet)
        {
            m_profileWallStartSeconds = wallNow;
            m_profileWallStartSet = true;
        }
        m_profileWallTimes[m_profileCount] = wallNow - m_profileWallStartSeconds;
        if (sampleCount > 0)
        {
            const Real scale =
                static_cast<Real>(m_tetIndices.size()) / static_cast<Real>(sampleCount);
            m_profileRotationMs[m_profileCount] = rotationAccum * scale;
            m_profileFaceMs[m_profileCount] = faceAccum * scale;
        }
        else
        {
            m_profileRotationMs[m_profileCount] = Real(0.0);
            m_profileFaceMs[m_profileCount] = Real(0.0);
        }
        m_profileSampleCounts[m_profileCount] = sampleCount;
        ++m_profileCount;
        m_profileLastTime = simTime;

        if ((simTime + dtSim) >= m_profileWindowValue - profileEps || m_profileCount >= m_profileCapacity)
        {
            m_profileDone = true;
            m_profileWritePending = true;
        }
    }

    f.endEdit();
}

void MagneticTetraForceField::handleEvent(sofa::core::objectmodel::Event* event)
{
    Inherited::handleEvent(event);

    if (!m_profileWritePending)
    {
        return;
    }
    m_profileWritePending = false;

    if (m_profileOutputPath.empty() || m_profileCount == 0)
    {
        return;
    }

    std::ofstream out(m_profileOutputPath, std::ios::out | std::ios::trunc);
    if (!out)
    {
        msg_warning() << "Failed to write profile CSV to " << m_profileOutputPath;
        return;
    }

    out << "wall_time_s,sim_time,addForce_ms,rotation_ms,face_ms,sample_count,sample_stride\n";
    for (size_t i = 0; i < m_profileCount; ++i)
    {
        out << m_profileWallTimes[i] << ","
            << m_profileSimTimes[i] << ","
            << m_profileDurationsMs[i] << ","
            << m_profileRotationMs[i] << ","
            << m_profileFaceMs[i] << ","
            << m_profileSampleCounts[i] << ","
            << m_profileSampleStride << "\n";
    }
    msg_info() << "MagneticTetraForceField profile saved to " << m_profileOutputPath;
}

void MagneticTetraForceField::addDForce(const sofa::core::MechanicalParams* mparams,
                                        DataVecDeriv& df,
                                        const DataVecDeriv& dx)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(df);
    SOFA_UNUSED(dx);
}

void MagneticTetraForceField::addKToMatrix(sofa::linearalgebra::BaseMatrix* /*matrix*/,
                                           SReal /*kFact*/,
                                           unsigned int& /*offset*/)
{
    // Magnetic force is treated explicitly; no stiffness contribution.
}

SReal MagneticTetraForceField::getPotentialEnergy(const sofa::core::MechanicalParams* mparams,
                                                  const DataVecCoord& x) const
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(x);
    return 0.0;
}

} // namespace magneticplugin

