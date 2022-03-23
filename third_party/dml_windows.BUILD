# Description:
#   Latest DirectML for onnxruntime

cc_library(
    name = "directml",
    hdrs = glob(["include/*.h"]),
    includes = ["include/"],
    data = ["bin/x64-win/DirectML.pdb"],
    defines = ["DML_TARGET_VERSION_USE_LATEST"],
    deps = [":dml_import_shared"],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "dml_import_shared",
    interface_library = "bin/x64-win/DirectML.lib",
    shared_library = "bin/x64-win/DirectML.dll",
    hdrs = glob(["include/*.h"]),
)