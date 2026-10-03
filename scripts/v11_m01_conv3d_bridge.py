"""Conv3d->Conv2d bridge for the VideoMAE tubelet patch embedding on torch_npu.

aclnnConvolutionBackward is not implemented for the tubelet Conv3d
(kernel=(2,16,16), stride=(2,16,16)) on this stack; Conv2d backward works.
For tubelet_size=2 the Conv3d is EXACTLY equivalent to stacking adjacent
frame pairs along the channel axis and applying a Conv2d:

  w2[o, k_t*3 + c, kh, kw] = w3[o, c, k_t, kh, kw]

Output keeps the Conv3d layout [B, D, T', H', W'] so PatchEmbed.forward
(proj -> flatten(2) -> transpose) works unchanged.
"""
import torch


class Conv3dBridge(torch.nn.Module):
    def __init__(self, conv3d: torch.nn.Conv3d):
        super().__init__()
        k_t = conv3d.kernel_size[0]
        if k_t != 2 or conv3d.stride != conv3d.kernel_size \
                or conv3d.padding != (0, 0, 0) or conv3d.dilation != (1, 1, 1) \
                or conv3d.groups != 1:
            raise ValueError(f'unsupported Conv3d config: k={conv3d.kernel_size} '
                             f's={conv3d.stride} p={conv3d.padding}')
        bias = conv3d.bias is not None
        oc, ic, _, kh, kw = conv3d.weight.shape
        self.conv2d = torch.nn.Conv2d(ic * k_t, oc, (kh, kw), stride=(kh, kw), bias=bias)
        with torch.no_grad():
            # x-side channel order after view/permute is (c, k_t) -> c*2+k_t,
            # which is exactly the natural flattening of [oc, ic, kt, kh, kw]
            self.conv2d.weight.copy_(conv3d.weight.reshape(oc, ic * k_t, kh, kw))
            if bias:
                self.conv2d.bias.copy_(conv3d.bias)

    def forward(self, x):  # x: [B, C, T, H, W], even T
        B, C, T, H, W = x.shape
        x = x.view(B, C, T // 2, 2, H, W).permute(0, 2, 1, 3, 4, 5).reshape(B * (T // 2), C * 2, H, W)
        y = self.conv2d(x)
        return y.view(B, T // 2, y.shape[1], y.shape[-2], y.shape[-1]).permute(0, 2, 1, 3, 4)


def patch_videomae_conv3d(vit) -> None:
    """Swap vit.patch_embed.proj (nn.Conv3d) for the bridge, in place."""
    vit.patch_embed.proj = Conv3dBridge(vit.patch_embed.proj)


if __name__ == '__main__':
    # CPU equivalence self-check against the original Conv3d
    torch.manual_seed(7)
    c3 = torch.nn.Conv3d(3, 768, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=False)
    bridge = Conv3dBridge(c3)
    x = torch.randn(2, 3, 16, 224, 224)
    with torch.no_grad():
        y3 = c3(x)
        yb = bridge(x)
    err = (y3 - yb).abs().max().item()
    # and gradient equivalence
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    c3(x1).sum().backward()
    bridge(x2).sum().backward()
    gerr = (x1.grad - x2.grad).abs().max().item()
    print(f'forward max abs diff: {err:.3e}  grad max abs diff: {gerr:.3e}')
    assert err < 1e-4 and gerr < 1e-4, 'bridge NOT equivalent'
    print('bridge equivalent: PASS')
