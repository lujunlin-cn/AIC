#ifndef _GDN_DECODE_STEP_KERNELS_H_
#define _GDN_DECODE_STEP_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class GdnDecodeStepCpuKernel : public CpuKernel {
public:
    ~GdnDecodeStepCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu
#endif
