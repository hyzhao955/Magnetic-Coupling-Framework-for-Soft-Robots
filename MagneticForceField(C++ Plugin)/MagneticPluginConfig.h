#pragma once

#include <sofa/config.h>

#ifdef SOFA_BUILD_MAGNETICPLUGIN
#  define SOFA_TARGET MagneticPlugin
#  define SOFA_MAGNETICPLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MAGNETICPLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace magneticplugin
{
    constexpr const char* MODULE_NAME = "MagneticPlugin";
    constexpr const char* MODULE_VERSION = "1.0";
} // namespace magneticplugin
