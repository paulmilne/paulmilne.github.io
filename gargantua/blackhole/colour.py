"""Blackbody colours and tone mapping.

Colour matching uses the multi-lobe Gaussian fits to the CIE 1931 2-degree
observer of Wyman, Sloan and Shirley (2013), JCGT 2(2), and the standard
XYZ to linear-sRGB matrix (D65 white).
"""
import numpy as np

H, C, KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23

_XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                     [-0.9689, 1.8758, 0.0415],
                     [0.0557, -0.2040, 1.0570]])


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cmf(lam_nm):
    """CIE 1931 colour-matching functions xbar, ybar, zbar at wavelengths in nm."""
    l = np.asarray(lam_nm, dtype=float)
    x = 1.056 * _g(l, 599.8, 37.9, 31.0) + 0.362 * _g(l, 442.0, 16.0, 26.7) - 0.065 * _g(l, 501.1, 20.4, 26.2)
    y = 0.821 * _g(l, 568.8, 46.9, 40.5) + 0.286 * _g(l, 530.9, 16.3, 31.1)
    z = 1.217 * _g(l, 437.0, 11.8, 36.0) + 0.681 * _g(l, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


def planck(lam_nm, T):
    """Spectral radiance B_lambda(T), W sr^-1 m^-3."""
    lam = np.asarray(lam_nm) * 1e-9
    with np.errstate(over="ignore"):
        return 2 * H * C ** 2 / lam ** 5 / np.expm1(H * C / (lam * KB * np.asarray(T)))


_LAM = np.linspace(360.0, 830.0, 471)
_CMF = cmf(_LAM)


def blackbody_xyz(T):
    """CIE XYZ of a blackbody at temperature(s) T, in arbitrary but
    consistent units (so relative brightness between temperatures is right)."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    B = planck(_LAM[None, :], T[:, None])            # (n, nlam)
    return B @ _CMF * (_LAM[1] - _LAM[0]) * 1e-9


class BlackbodyTable:
    """Fast lookup of blackbody XYZ on a log-spaced temperature grid."""

    def __init__(self, t_min=500.0, t_max=1e5, n=2000):
        self.logT = np.linspace(np.log(t_min), np.log(t_max), n)
        xyz = blackbody_xyz(np.exp(self.logT))
        self.logxyz = np.log(np.maximum(xyz, 1e-300))

    def __call__(self, T):
        lt = np.log(np.clip(T, np.exp(self.logT[0]), np.exp(self.logT[-1])))
        out = np.stack([np.interp(lt, self.logT, self.logxyz[:, i]) for i in range(3)], -1)
        return np.where(np.asarray(T)[..., None] > 0, np.exp(out), 0.0)


def xyz_to_rgb(xyz):
    return xyz @ _XYZ2RGB.T


def chromaticity_rgb(T):
    """Linear sRGB of a blackbody normalised to unit luminance Y."""
    xyz = blackbody_xyz(T)
    rgb = xyz_to_rgb(xyz / xyz[:, 1:2])
    return np.clip(rgb, 0, None)


def tonemap(rgb, exposure=1.0, white=None):
    """Exposure, a filmic (ACES-style, Narkowicz 2015) curve and the sRGB
    transfer function. Out-of-gamut values are desaturated towards their
    luminance rather than clipped per channel."""
    x = np.asarray(rgb) * exposure
    lum = (x @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    neg = np.minimum(x.min(-1, keepdims=True), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(neg < 0, lum / (lum - neg + 1e-12), 1.0)
    x = lum + (x - lum) * np.clip(t, 0, 1)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def bloom(rgb, sigmas=(2.0, 6.0, 18.0), weights=(0.04, 0.03, 0.02)):
    """Add a soft glare around bright regions, a stand-in for the camera's
    optics (scattering in the lens), not part of the physics."""
    from scipy.ndimage import gaussian_filter
    out = rgb.copy()
    for s, w in zip(sigmas, weights):
        out += w * np.stack([gaussian_filter(rgb[..., i], s, mode="nearest") for i in range(3)], -1)
    return out
