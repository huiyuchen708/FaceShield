import os
import sys
import cv2
import json
import argparse
import numpy as np
from typing import Optional, Tuple
from PIL import Image

try:
    sys.path.insert(0, 'E://FaceShield')
    from preprocess.mtcnn import MTCNN
    mtcnn = MTCNN()
except Exception as _e:
    mtcnn = None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def align_face(image_pil: Image.Image, crop_size: int = 224, reverse: bool = True):
    if mtcnn is None:
        return None
    if reverse:
        out = mtcnn.align_multi(image_pil, min_face_size=64., thresholds=[0.6, 0.7, 0.7],
                                crop_size=(crop_size, crop_size), reverse=True)
        if out is None:
            mini = 20.
            th1, th2, th3 = 0.6, 0.6, 0.6
            while out is None and mini >= 5.:
                out = mtcnn.align_multi(image_pil, min_face_size=mini,
                                        thresholds=[th1, th2, th3],
                                        crop_size=(crop_size, crop_size), reverse=True)
                if out is None:
                    th1 *= 0.8
                    th2 *= 0.8
                    th3 *= 0.8
                    mini *= 0.8
        return out
    else:
        faces = mtcnn.align_multi(image_pil, min_face_size=64., thresholds=[0.6, 0.7, 0.8],
                                  factor=0.707, crop_size=(crop_size, crop_size))
        return faces


def choose_largest_face(boxes: Optional[np.ndarray]) -> int:
    if boxes is None or len(boxes) == 0:
        return 0
    best, best_area = 0, 0.0
    for i, box in enumerate(boxes):
        w = box[2] - box[0] + 1.0
        h = box[3] - box[1] + 1.0
        area = float(w * h)
        if area > best_area:
            best_area = area
            best = i
    return best


def compute_center_from_tfm_inv(tfm_inv: np.ndarray, crop_size: int) -> Tuple[float, float]:
    src_pt = np.array([crop_size / 2.0, crop_size / 2.0, 1.0], dtype=np.float32).reshape(3, 1)
    x, y = np.matmul(tfm_inv, src_pt)
    return float(x[0]), float(y[0])


def default_mask_rect_for_crop(crop_size: int = 224):
    if crop_size == 224:
        return [18, 206, 35, 189]
    scale = crop_size / 224.0
    y0, y1, x0, x1 = 18 * scale, 206 * scale, 35 * scale, 189 * scale
    return [int(round(y0)), int(round(y1)), int(round(x0)), int(round(x1))]


def save_metadata(meta_path: str,
                  original_path: str,
                  aligned_face_path: str,
                  H: int, W: int,
                  crop_size: int,
                  bbox: Optional[np.ndarray],
                  tfm_inv: Optional[np.ndarray]):
    meta = {
        'original_path': original_path,
        'aligned_face_path': aligned_face_path,
        'original_size': {'height': int(H), 'width': int(W)},
        'crop_size': int(crop_size),
        'bbox': bbox.tolist() if bbox is not None else None,
        'tfm_inv': tfm_inv.tolist() if tfm_inv is not None else None,
        'mask_rect_aligned': default_mask_rect_for_crop(crop_size),  # [y0,y1,x0,x1]
    }
    if tfm_inv is not None:
        cx, cy = compute_center_from_tfm_inv(np.asarray(tfm_inv, dtype=np.float32), crop_size)
        meta['seamless_center'] = {'x': cx, 'y': cy}
    ensure_dir(os.path.dirname(meta_path))
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def process_image(image_path: str, out_root: str, crop_size: int = 224) -> bool:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return False

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    out = align_face(img_pil, crop_size=crop_size, reverse=True)
    if out is None:
        return False

    faces, tfm_invs, boxes = out
    fi = choose_largest_face(boxes)
    aligned_pil: Image.Image = faces[fi]
    tfm_inv: np.ndarray = tfm_invs[fi]
    bbox = boxes[fi] if boxes is not None else None
    abs_input = os.path.abspath(image_path)
    faces_root = os.path.join(out_root, 'faces')
    meta_root = os.path.join(out_root, 'meta')
    ensure_dir(faces_root)
    ensure_dir(meta_root)

    rel_path = os.path.basename(abs_input)
    base_name, _ = os.path.splitext(rel_path)
    face_save_path = os.path.join(faces_root, f"{base_name}_aligned_{crop_size}.png")
    meta_save_path = os.path.join(meta_root, f"{base_name}_aligned_{crop_size}.json")

    aligned_rgb = np.array(aligned_pil)
    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
    ensure_dir(os.path.dirname(face_save_path))
    cv2.imwrite(face_save_path, aligned_bgr)

    save_metadata(meta_save_path, abs_input, face_save_path, H, W, crop_size, bbox, tfm_inv)
    return True


def process_image_with_rel(image_path: str, input_root: str, out_root: str, crop_size: int = 224) -> bool:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return False

    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    out = align_face(img_pil, crop_size=crop_size, reverse=True)
    if out is None:
        return False

    faces, tfm_invs, boxes = out
    fi = choose_largest_face(boxes)
    aligned_pil: Image.Image = faces[fi]
    tfm_inv: np.ndarray = tfm_invs[fi]
    bbox = boxes[fi] if boxes is not None else None

    rel_path = os.path.relpath(os.path.abspath(image_path), os.path.abspath(input_root))
    rel_dir = os.path.dirname(rel_path)
    stem, _ = os.path.splitext(os.path.basename(rel_path))

    faces_root = os.path.join(out_root, 'faces', rel_dir)
    meta_root = os.path.join(out_root, 'meta', rel_dir)
    ensure_dir(faces_root)
    ensure_dir(meta_root)

    face_save_path = os.path.join(faces_root, f"{stem}_aligned_{crop_size}.png")
    meta_save_path = os.path.join(meta_root, f"{stem}_aligned_{crop_size}.json")

    aligned_rgb = np.array(aligned_pil)
    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(face_save_path, aligned_bgr)

    save_metadata(meta_save_path, os.path.abspath(image_path), face_save_path, H, W, crop_size, bbox, tfm_inv)
    return True


def is_image_file(name: str) -> bool:
    name = name.lower()
    return name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--input', type=str, required=True, help='Input file or directory')
    p.add_argument('-o', '--out_dir', type=str, default='E://FaceShield/aligned_cache', help='Output cache directory')
    p.add_argument('--crop_size', type=int, default=224, help='The size of the aligned face')
    args = p.parse_args()

    if mtcnn is None:
        return

    out_root = os.path.abspath(args.out_dir)
    ensure_dir(out_root)

    input_path = os.path.abspath(args.input)
    crop = int(args.crop_size)

    total, failed = 0, 0
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for fname in files:
                if not is_image_file(fname):
                    continue
                fpath = os.path.join(root, fname)
                ok = process_image_with_rel(fpath, input_path, out_root, crop_size=crop)
                total += 1
                if not ok:
                    failed += 1
    else:
        ok = process_image(input_path, out_root, crop_size=crop)
        total = 1
        failed = 0 if ok else 1

if __name__ == '__main__':
    main()


