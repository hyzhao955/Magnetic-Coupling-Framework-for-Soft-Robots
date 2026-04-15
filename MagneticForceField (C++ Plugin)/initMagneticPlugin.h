#pragma once

#include "MagneticPluginConfig.h"

// C-style plugin entry points exposed to the SOFA loader.
// 向 SOFA 加载器暴露的 C 风格插件入口。
extern "C" {
SOFA_MAGNETICPLUGIN_API void initExternalModule();
SOFA_MAGNETICPLUGIN_API const char* getModuleName();
SOFA_MAGNETICPLUGIN_API const char* getModuleVersion();
SOFA_MAGNETICPLUGIN_API const char* getModuleLicense();
SOFA_MAGNETICPLUGIN_API const char* getModuleDescription();
SOFA_MAGNETICPLUGIN_API const char* getModuleComponentList();
SOFA_MAGNETICPLUGIN_API bool moduleIsInitialized();
}
