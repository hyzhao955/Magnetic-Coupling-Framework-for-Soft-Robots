#include "MagneticPluginConfig.h"
#include <sofa/core/ObjectFactory.h>

#include "initMagneticPlugin.h"
#include "MagneticTetraForceField.h"

using magneticplugin::MagneticTetraForceField;

namespace
{
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
    initExternalModule();
    return 0;
}

SOFA_LINK_CLASS(MagneticTetraForceField);
