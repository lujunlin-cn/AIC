#ifndef _GDN_DECODE_STEP_PROTO_H_
#define _GDN_DECODE_STEP_PROTO_H_

#include "graph/operator_reg.h"


namespace ge {
// decode 单步 GDN 递推(forward-only)
// q/k: [B, Hv, dk] fp16(已 l2norm+scale); v: [B, Hv, dv] fp16
// g: [B, Hv] fp32(log 空间 <=0); beta: [B, Hv] fp32
// state: [B, Hv, dk, dv] fp32
REG_OP(GdnDecodeStep)
    .INPUT(q, TensorType({DT_FLOAT16}))
    .INPUT(k, TensorType({DT_FLOAT16}))
    .INPUT(v, TensorType({DT_FLOAT16}))
    .INPUT(g, TensorType({DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT}))
    .INPUT(state, TensorType({DT_FLOAT}))
    .OUTPUT(out, TensorType({DT_FLOAT16}))
    .OUTPUT(new_state, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(GdnDecodeStep)
}  // namespace ge

#endif  // _GDN_DECODE_STEP_PROTO_H_
