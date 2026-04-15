#pragma once

#include <sofa/config.h>

// Export/import macros used when building or consuming the plugin.
// 构建或使用该插件时所用的导出/导入宏。
#ifdef SOFA_BUILD_MAGNETICPLUGIN
#  define SOFA_TARGET MagneticPlugin
#  define SOFA_MAGNETICPLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MAGNETICPLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace magneticplugin
{
    // Static metadata queried by the plugin loader.
    // 供插件加载器查询的静态元数据。
    constexpr const char* MODULE_NAME = "MagneticPlugin";
    constexpr const char* MODULE_VERSION = "1.0";
} // namespace magneticplugin
