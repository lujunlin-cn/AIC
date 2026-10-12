// GdnAddDemo 单算子执行验证(aclopExecuteV2): 期望 z = x + y
#include <acl/acl.h>
#include <acl/acl_op.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#define CHECK(fn)                                                            \
    do {                                                                     \
        aclError _e = (fn);                                                  \
        if (_e != ACL_SUCCESS) {                                             \
            printf("FAIL %s:%d %s err=%d\n", __FILE__, __LINE__, #fn, (int)_e); \
            return 1;                                                        \
        }                                                                    \
    } while (0)

int main(int argc, char **argv) {
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtContext ctx;
    CHECK(aclrtCreateContext(&ctx, 0));
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream));

    const int64_t n = 4;
    const float hx[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float hy[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float hz[4] = {0, 0, 0, 0};

    aclTensorDesc *desc = aclCreateTensorDesc(ACL_FLOAT, 1, &n, ACL_FORMAT_ND);
    void *dx = nullptr, *dy = nullptr, *dz = nullptr;
    CHECK(aclrtMalloc(&dx, n * sizeof(float), ACL_MEM_MALLOC_NORMAL));
    CHECK(aclrtMalloc(&dy, n * sizeof(float), ACL_MEM_MALLOC_NORMAL));
    CHECK(aclrtMalloc(&dz, n * sizeof(float), ACL_MEM_MALLOC_NORMAL));
    CHECK(aclrtMemcpy(dx, n * sizeof(float), hx, n * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(dy, n * sizeof(float), hy, n * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE));

    aclDataBuffer *bx = aclCreateDataBuffer(dx, n * sizeof(float));
    aclDataBuffer *by = aclCreateDataBuffer(dy, n * sizeof(float));
    aclDataBuffer *bz = aclCreateDataBuffer(dz, n * sizeof(float));
    aclTensorDesc *outs[1] = {desc};
    aclDataBuffer *bout[1] = {bz};
    aclTensorDesc *ins[2] = {desc, desc};
    aclDataBuffer *bin[2] = {bx, by};

    CHECK(aclopExecuteV2("GdnAddDemo", 2, ins, bin, 1, outs, bout, 0, nullptr,
                         stream));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(hz, sizeof(hz), dz, n * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST));

    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (std::fabs(hz[i] - (hx[i] + hy[i])) > 1e-6) {
            ok = 0;
        }
        printf("z[%d]=%f expect=%f\n", i, hz[i], hx[i] + hy[i]);
    }
    printf(ok ? "ADD_OK\n" : "ADD_WRONG\n");

    aclDestroyDataBuffer(bx);
    aclDestroyDataBuffer(by);
    aclDestroyDataBuffer(bz);
    aclDestroyTensorDesc(desc);
    aclrtFree(dx);
    aclrtFree(dy);
    aclrtFree(dz);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();
    return ok ? 0 : 1;
}
