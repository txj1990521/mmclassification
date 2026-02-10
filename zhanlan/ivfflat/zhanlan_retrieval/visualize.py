# zhanlan/ivfflat/zhanlan_retrieval/visualize.py
import cv2
import numpy as np
import math


def put_text(img, text, org=(8, 26), font_scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def fit_square(img_bgr, tile=320):
    h, w = img_bgr.shape[:2]
    scale = tile / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
    y0 = (tile - nh) // 2
    x0 = (tile - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def visualize_grid(query_bgr, top_imgs_bgr, top_scores, out_path,
                   tile=320, gap=10, header=44):
    tiles = [fit_square(query_bgr, tile)]
    labels = ["QUERY"]

    for i, (img, s) in enumerate(zip(top_imgs_bgr, top_scores), 1):
        if img is None:
            img = np.zeros((tile, tile, 3), np.uint8)
        tiles.append(fit_square(img, tile))
        labels.append(f"#{i}  {float(s):.3f}")

    n = len(tiles)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    H = rows * (tile + header) + (rows + 1) * gap
    W = cols * tile + (cols + 1) * gap
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for idx in range(n):
        r = idx // cols
        c = idx % cols
        x = gap + c * (tile + gap)
        y = gap + r * (tile + header + gap)

        canvas[y:y+header, x:x+tile] = 0
        put_text(canvas, labels[idx], org=(x + 8, y + 28), font_scale=0.7, thickness=2)
        canvas[y+header:y+header+tile, x:x+tile] = tiles[idx]

    cv2.imwrite(out_path, canvas)
    return out_path
