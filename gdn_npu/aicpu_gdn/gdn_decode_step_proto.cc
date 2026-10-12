#include "gdn_decode_step_proto.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(GdnDecodeStepInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(GdnDecodeStep, GdnDecodeStepVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(GdnDecodeStep, GdnDecodeStepInferShape);
VERIFY_FUNC_REG(GdnDecodeStep, GdnDecodeStepVerify);
}  // namespace ge
