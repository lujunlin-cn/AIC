// GdnDecodeStep AICPU kernel: decode 单步 GDN 递推融合(forward-only).
//
// 数学(与 qwen35_gdn_npu.npu_recurrent_gated_delta_rule 单步完全一致):
//   q/k: [B,Hv,dk] fp16(已 l2norm+1/sqrt(dk)); v: [B,Hv,dv] fp16
//   g: [B,Hv] fp32 log 空间; beta: [B,Hv] fp32; state: [B,Hv,dk,dv] fp32
//   per (b,h):
//     state *= exp(g)
//     u      = state^T k          (dv)
//     delta  = (v - u) * beta
//     state += k (x) delta
//     out    = state^T q          (dv)
// 输出: out [B,Hv,dv] fp16; new_state [B,Hv,dk,dv] fp32
#include "gdn_decode_step_kernels.h"

#include "cpu_kernel.h"

#include <cmath>
#include <cstdint>

namespace aicpu {
namespace {
inline float half2float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t man = h & 0x03FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x0400) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x03FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t float2half(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t man = bits & 0x007FFFFF;
    if (((bits >> 23) & 0xFF) == 0xFF) {
        return static_cast<uint16_t>(sign | 0x7C00 | (man ? 0x0200 : 0));
    }
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        man = (man | 0x00800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | (man >> 13));
    }
    return static_cast<uint16_t>(sign | (exp << 10) | (man >> 13));
}
}  // namespace

uint32_t GdnDecodeStepCpuKernel::Compute(CpuKernelContext &ctx) {
    Tensor *tq = ctx.Input(0);
    Tensor *tk = ctx.Input(1);
    Tensor *tv = ctx.Input(2);
    Tensor *tg = ctx.Input(3);
    Tensor *tb = ctx.Input(4);
    Tensor *ts = ctx.Input(5);
    Tensor *tout = ctx.Output(0);
    Tensor *tstate = ctx.Output(1);
    if (!tq || !tk || !tv || !tg || !tb || !ts || !tout || !tstate) {
        return 1;
    }
    if (tq->GetDataType() != DT_FLOAT16 || tk->GetDataType() != DT_FLOAT16 ||
        tv->GetDataType() != DT_FLOAT16 || ts->GetDataType() != DT_FLOAT) {
        return 2;
    }

    // shape 推导: q [B,Hv,dk], v [B,Hv,dv], state [B,Hv,dk,dv]
    const int64_t B = tq->GetTensorShape()->GetDimSize(0);
    const int64_t Hv = tq->GetTensorShape()->GetDimSize(1);
    const int64_t dk = tq->GetTensorShape()->GetDimSize(2);
    const int64_t dv = tv->GetTensorShape()->GetDimSize(2);

    const uint16_t *qp = static_cast<const uint16_t *>(tq->GetData());
    const uint16_t *kp = static_cast<const uint16_t *>(tk->GetData());
    const uint16_t *vp = static_cast<const uint16_t *>(tv->GetData());
    const float *gp = static_cast<const float *>(tg->GetData());
    const float *bp = static_cast<const float *>(tb->GetData());
    const float *sp = static_cast<const float *>(ts->GetData());
    uint16_t *op = static_cast<uint16_t *>(tout->GetData());
    float *osp = static_cast<float *>(tstate->GetData());

    // 主循环: 每 (b,h) 一个 dk x dv 状态的单步递推, fp32 累加
    for (int64_t bh = 0; bh < B * Hv; ++bh) {
        const float decay = std::exp(gp[bh]);
        const float beta = bp[bh];
        const uint16_t *q = qp + bh * dk;
        const uint16_t *k = kp + bh * dk;
        const uint16_t *v = vp + bh * dv;
        const float *s_in = sp + bh * dk * dv;
        float *s = osp + bh * dk * dv;
        uint16_t *o = op + bh * dv;

        // state = decay * state_in
        for (int64_t i = 0; i < dk * dv; ++i) {
            s[i] = decay * s_in[i];
        }
        // u[dv] = sum_dk s[dk][dv] * k[dk]
        float u[512];
        for (int64_t j = 0; j < dv; ++j) {
            float acc = 0.0f;
            for (int64_t i = 0; i < dk; ++i) {
                acc += s[i * dv + j] * half2float(k[i]);
            }
            u[j] = acc;
        }
        // delta[dv] = (v - u) * beta
        float delta[512];
        for (int64_t j = 0; j < dv; ++j) {
            delta[j] = (half2float(v[j]) - u[j]) * beta;
        }
        // s[dk][dv] += k[dk] * delta[dv]; 同时 out[dv] = sum_dk s[dk][dv] * q[dk]
        float out_acc[512];
        for (int64_t j = 0; j < dv; ++j) {
            out_acc[j] = 0.0f;
        }
        for (int64_t i = 0; i < dk; ++i) {
            const float kv = half2float(k[i]);
            const float qv = half2float(q[i]);
            float *row = s + i * dv;
            for (int64_t j = 0; j < dv; ++j) {
                row[j] += kv * delta[j];
                out_acc[j] += row[j] * qv;
            }
        }
        for (int64_t j = 0; j < dv; ++j) {
            o[j] = float2half(out_acc[j]);
        }
    }
    return 0;
}

const char *GDN_DECODE_STEP = "GdnDecodeStep";
REGISTER_CPU_KERNEL(GDN_DECODE_STEP, GdnDecodeStepCpuKernel);
}  // namespace aicpu
