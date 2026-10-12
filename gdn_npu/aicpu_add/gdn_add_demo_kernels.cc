// GdnAddDemo AICPU 实现: z = x + y (fp32/fp16)
#include "gdn_add_demo_kernels.h"

#include "cpu_kernel.h"
#include <cstdint>

namespace aicpu {
namespace {
// IEEE754 half<->float 位级转换(AICPU 环境无 Eigen half 依赖)
inline float half2float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t man = h & 0x03FF;
    uint32_t bits;
    if (exp == 0) {  // subnormal/zero
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
    } else if (exp == 0x1F) {  // inf/nan
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
    if (((bits >> 23) & 0xFF) == 0xFF) {  // inf/nan
        return static_cast<uint16_t>(sign | 0x7C00 | (man ? 0x0200 : 0));
    }
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00);  // overflow -> inf
    }
    if (exp <= 0) {  // subnormal/zero(近端舍入从简)
        if (exp < -10) {
            return sign;
        }
        man = (man | 0x00800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | (man >> 13));
    }
    return static_cast<uint16_t>(sign | (exp << 10) | (man >> 13));
}
}  // namespace

uint32_t GdnAddDemoCpuKernel::Compute(CpuKernelContext &ctx) {
    Tensor *in0 = ctx.Input(0);
    Tensor *in1 = ctx.Input(1);
    Tensor *out = ctx.Output(0);
    if (in0 == nullptr || in1 == nullptr || out == nullptr) {
        return 1;
    }
    const int64_t n = in0->NumElements();
    if (in1->NumElements() != n || out->NumElements() != n) {
        return 2;
    }
    const DataType dt = in0->GetDataType();
    if (dt == DT_FLOAT) {
        const float *x = static_cast<const float *>(in0->GetData());
        const float *y = static_cast<const float *>(in1->GetData());
        float *z = static_cast<float *>(out->GetData());
        for (int64_t i = 0; i < n; i++) {
            z[i] = x[i] + y[i];
        }
    } else if (dt == DT_FLOAT16) {
        const uint16_t *x = static_cast<const uint16_t *>(in0->GetData());
        const uint16_t *y = static_cast<const uint16_t *>(in1->GetData());
        uint16_t *z = static_cast<uint16_t *>(out->GetData());
        for (int64_t i = 0; i < n; i++) {
            z[i] = float2half(half2float(x[i]) + half2float(y[i]));
        }
    } else {
        return 3;  // 不支持的 dtype
    }
    return 0;
}

const char *GDN_ADD_DEMO = "GdnAddDemo";
REGISTER_CPU_KERNEL(GDN_ADD_DEMO, GdnAddDemoCpuKernel);
}  // namespace aicpu
