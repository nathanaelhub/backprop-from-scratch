"""Visualizations for backprop.py — still zero dependencies.

Python's stdlib ships zlib and struct, which is everything a minimal PNG
encoder needs (8-bit RGB, filter type 0). So in keeping with the project's
from-scratch ethos, this module hand-rolls the image format too and renders:

    assets/boundary_xor.png       learned decision boundary, noisy XOR
    assets/boundary_clusters.png  learned decision boundary, 3-class blobs
    assets/loss_curves.png        cross-entropy loss vs. training progress

Run:  python3 visualize.py
"""

import math
import os
import struct
import zlib

from backprop import MLP, cross_entropy, make_clusters, make_xor

WHITE = (255, 255, 255)
GRID_GRAY = (229, 231, 235)
AXIS_GRAY = (107, 114, 128)
# Class palette: violet, emerald, amber.
PALETTE = [(124, 58, 237), (16, 185, 129), (245, 158, 11)]


# ----------------------------------------------------------------------------
# Minimal PNG encoder (spec: PNG signature + IHDR / IDAT / IEND chunks,
# IDAT holds zlib-compressed scanlines, each prefixed with filter byte 0)
# ----------------------------------------------------------------------------

def write_png(path, pixels):
    """pixels: list of rows, each row a list of (r, g, b) tuples."""
    h, w = len(pixels), len(pixels[0])
    raw = b"".join(
        b"\x00" + bytes(v for px in row for v in px) for row in pixels
    )

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)
    print(f"  wrote {path}  ({w}x{h}, {len(png)} bytes)")


# ----------------------------------------------------------------------------
# Drawing primitives
# ----------------------------------------------------------------------------

def blank(w, h, color=WHITE):
    return [[color] * w for _ in range(h)]


def draw_disc(img, cx, cy, r, color):
    h, w = len(img), len(img[0])
    for y in range(int(cy - r), int(cy + r) + 1):
        for x in range(int(cx - r), int(cx + r) + 1):
            if 0 <= y < h and 0 <= x < w and (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                img[y][x] = color


def draw_line(img, x0, y0, x1, y1, color, thick=1):
    """Bresenham line, optionally thickened with small discs."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        if thick > 1:
            draw_disc(img, x0, y0, thick / 2, color)
        elif 0 <= y0 < len(img) and 0 <= x0 < len(img[0]):
            img[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def mix(c1, c2, t):
    """Linear blend: t=0 -> c1, t=1 -> c2."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ----------------------------------------------------------------------------
# Training (quiet — same loop as backprop.train, without the console output)
# ----------------------------------------------------------------------------

def fit(X, y, n_classes, hidden, lr, epochs):
    net = MLP(d=len(X[0]), h=hidden, k=n_classes)
    losses = []
    for _ in range(epochs):
        P, cache = net.forward(X)
        losses.append(cross_entropy(P, y))
        net.step(net.backward(P, y, cache), lr)
    return net, losses


# ----------------------------------------------------------------------------
# Decision boundary plot
# ----------------------------------------------------------------------------

def boundary_png(path, net, X, y, grid_w=220, grid_h=165, upscale=3):
    xs, ys = [p[0] for p in X], [p[1] for p in X]
    mx, my = (max(xs) - min(xs)) * 0.18, (max(ys) - min(ys)) * 0.18
    x_lo, x_hi = min(xs) - mx, max(xs) + mx
    y_lo, y_hi = min(ys) - my, max(ys) + my

    # Background: classify every grid cell, color = probability-weighted
    # blend of the class palette, washed toward white so points stay legible.
    bg = []
    for gy in range(grid_h):
        # Image rows run top-down; data y runs bottom-up.
        wy = y_hi - (gy + 0.5) / grid_h * (y_hi - y_lo)
        batch = [[x_lo + (gx + 0.5) / grid_w * (x_hi - x_lo), wy]
                 for gx in range(grid_w)]
        P, _ = net.forward(batch)
        row = []
        for p in P:
            blend = tuple(
                int(sum(p[c] * PALETTE[c][ch] for c in range(len(p))))
                for ch in range(3)
            )
            row.append(mix(blend, WHITE, 0.62))
        bg.append(row)

    # Nearest-neighbour upscale, then overlay the training points.
    img = [[bg[gy][gx] for gx in range(grid_w) for _ in range(upscale)]
           for gy in range(grid_h) for _ in range(upscale)]
    w, h = grid_w * upscale, grid_h * upscale
    for (px, py), label in zip(X, y):
        cx = (px - x_lo) / (x_hi - x_lo) * w
        cy = (y_hi - py) / (y_hi - y_lo) * h
        draw_disc(img, cx, cy, 4.5, WHITE)
        draw_disc(img, cx, cy, 3.2, PALETTE[label])

    write_png(path, img)


# ----------------------------------------------------------------------------
# Loss curve plot
# ----------------------------------------------------------------------------

def loss_curves_png(path, series, w=660, h=330):
    """series: list of (label, losses, color). X axis is normalized training
    progress so runs with different epoch counts share one chart."""
    img = blank(w, h)
    ml, mr, mt, mb = 46, 14, 14, 26  # margins around the plot area
    pw, ph = w - ml - mr, h - mt - mb
    hi = max(max(s[1]) for s in series)

    # Horizontal gridlines at quarter intervals of the loss range.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = mt + int(ph * (1 - frac))
        draw_line(img, ml, gy, ml + pw, gy, GRID_GRAY)
    draw_line(img, ml, mt, ml, mt + ph, AXIS_GRAY)
    draw_line(img, ml, mt + ph, ml + pw, mt + ph, AXIS_GRAY)

    for _, losses, color in series:
        pts = [
            (ml + i / (len(losses) - 1) * pw, mt + (1 - v / hi) * ph)
            for i, v in enumerate(losses)
        ]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            draw_line(img, x0, y0, x1, y1, color, thick=3)

    # Legend swatches, top-right.
    for i, (label, _, color) in enumerate(series):
        draw_disc(img, ml + pw - 150, mt + 18 + i * 18, 5, color)
        print(f"  legend: {label}")
    write_png(path, img)


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(out, exist_ok=True)

    print("Training XOR network...")
    X1, y1, k1 = make_xor()
    net1, loss1 = fit(X1, y1, k1, hidden=8, lr=0.5, epochs=600)

    print("Training cluster network...")
    X2, y2, k2 = make_clusters()
    net2, loss2 = fit(X2, y2, k2, hidden=8, lr=0.3, epochs=300)

    print("Rendering PNGs (pure stdlib: zlib + struct)...")
    boundary_png(os.path.join(out, "boundary_xor.png"), net1, X1, y1)
    boundary_png(os.path.join(out, "boundary_clusters.png"), net2, X2, y2)
    loss_curves_png(
        os.path.join(out, "loss_curves.png"),
        [("XOR", loss1, PALETTE[0]), ("Clusters", loss2, PALETTE[1])],
    )


if __name__ == "__main__":
    main()
