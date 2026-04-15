#include "MagneticPluginConfig.h"
#include <sofa/core/ObjectFactory.h>

#include "initMagneticPlugin.h"
#include "MagneticTetraForceField.h"

using magneticplugin::MagneticTetraForceField;

namespace
{
// Register plugin objects only once.
// 仅注册一次插件对象。
bool g_isInitialized = false;
}

extern "C"
{
SOFA_MAGNETICPLUGIN_API void initExternalModule()
{
    if (g_isInitialized)
    {
        return;
    }
    g_isInitialized = true;

    // Register the custom force field in SOFA's object factory.
    // 在 SOFA 对象工厂中注册这个自定义力场。
    static int MagneticTetraForceFieldClass =
        sofa::core::RegisterObject("Magnetic force field for tetrahedral meshes")
            .add<MagneticTetraForceField>();
    (void)MagneticTetraForceFieldClass;
}

SOFA_MAGNETICPLUGIN_API const char* getModuleName()
{
    return magneticplugin::MODULE_NAME;
}

SOFA_MAGNETICPLUGIN_API const char* getModuleDescription()
{
    return "Magnetic force field for tetrahedral meshes";
}

SOFA_MAGNETICPLUGIN_API const char* getModuleLicense()
{
    return "LGPL";
}

SOFA_MAGNETICPLUGIN_API const char* getModuleVersion()
{
    return magneticplugin::MODULE_VERSION;
}

SOFA_MAGNETICPLUGIN_API const char* getModuleComponentList()
{
    return "";
}

SOFA_MAGNETICPLUGIN_API bool moduleIsInitialized()
{
    return g_isInitialized;
}
}

int MagneticPlugin_init()
{
    // Compatibility entry point for explicit plugin loading.
    // 用于显式加载插件的兼容入口。
    initExternalModule();
    return 0;
}

SOFA_LINK_CLASS(MagneticTetraForceField);
