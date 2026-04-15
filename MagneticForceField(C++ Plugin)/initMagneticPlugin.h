#pragma once

#include "MagneticPluginConfig.h"

extern "C" {
SOFA_MAGNETICPLUGIN_API void initExternalModule();
SOFA_MAGNETICPLUGIN_API const char* getModuleName();
SOFA_MAGNETICPLUGIN_API const char* getModuleVersion();
SOFA_MAGNETICPLUGIN_API const char* getModuleLicense();
SOFA_MAGNETICPLUGIN_API const char* getModuleDescription();
SOFA_MAGNETICPLUGIN_API const char* getModuleComponentList();
SOFA_MAGNETICPLUGIN_API bool moduleIsInitialized();
}
