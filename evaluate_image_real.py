from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import requests
import numpy as np
from PIL import Image
import torchvision.transforms as T
from threading import Thread, Lock
import time
import threading

from flask import Flask, send_file, jsonify
import io
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Tuple, Optional




import torch
import math


app = Flask(__name__)

window = False

# https://github.com/IDEA-Research/GroundingDINO
baseHost = "http://192.168.0.84"
url = f"{baseHost}/cam-hi.jpg"

def fetch_image_from_url(url):
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def preprocess_image_from_array(image_np: np.ndarray): #-> Tuple[np.ndarray, torch.Tensor]:
    transform = T.Compose(
        [
            T.Resize(800),  # Optional: match original RandomResize behavior
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Convert NumPy BGR (OpenCV) to RGB and then to PIL
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    image_transformed = transform(image_pil)
    return image_np, image_transformed

def postprocess_masks(
    masks: torch.Tensor,
    original_size: Tuple[int, int],
    input_size: Tuple[int, int],
    crop_box: Optional[Tuple[int,int,int,int]] = None,
    threshold: float = 0.5,
) -> List[np.ndarray]:
    """
    1) If masks are from a cropped window, paste them back into a full-size canvas
       of shape input_size.
    2) Resize from input_size (H_pad, W_pad) → original_size (raw_h, raw_w).
    3) Binarize at `threshold`.

    Args:
      masks:         torch.Tensor with shape (N, Hm, Wm) or (1,N,Hm,Wm), etc.
      original_size: (raw_h, raw_w)
      input_size:    (H_pad, W_pad) that SAM internally used
      crop_box:      (x0, y0, x1, y1) coordinates of the crop within the padded image.
                     If None, we assume masks already cover the full (H_pad, W_pad).
      threshold:     probability/logit cutoff to make a binary mask.

    Returns:
      List of N uint8 numpy masks of shape (raw_h, raw_w)
    """
    # → 1) numpy + collapse leading dims
    masks_np = masks.detach().cpu().numpy()
    *lead, Hm, Wm = masks_np.shape
    N = int(np.prod(lead))
    masks_np = masks_np.reshape(N, Hm, Wm)

    raw_h, raw_w = map(int, original_size)
    pad_h, pad_w = map(int, input_size)

    out_masks = []
    for i, m in enumerate(masks_np):
        # If this mask is from a crop, paste it back
        if crop_box and (Hm, Wm) != (pad_h, pad_w):
            x0, y0, x1, y1 = crop_box
            # ensure box dims match mask dims
            assert (y1-y0 == Hm) and (x1-x0 == Wm), (
                f"Crop box {(x0,y0,x1,y1)} size != mask shape {(Hm, Wm)}"
            )
            full_mask = np.zeros((pad_h, pad_w), dtype=np.float32)
            full_mask[y0:y1, x0:x1] = m.astype(np.float32)
        else:
            full_mask = m.astype(np.float32)

        # Resize from padded → original
        resized = cv2.resize(
            full_mask,
            dsize=(raw_w, raw_h),           # OpenCV wants (width, height)
            interpolation=cv2.INTER_NEAREST  # preserve crisp edges
        )

        # Binarize
        bin_mask = (resized > threshold).astype(np.uint8)
        out_masks.append(bin_mask)

    return out_masks

def overlay_masks_cv2(
    image: np.ndarray,
    masks: List[np.ndarray],
    color: Tuple[int, int, int] = (30, 144, 255),
    alpha: float = 0.6
) -> np.ndarray:
    """
    Overlay one or more binary masks onto an image with OpenCV.

    Args:
        image:  H×W×3 BGR uint8 source image.
        masks:  List of H×W binary uint8 masks (0 or 1).
        color:  BGR tuple for the mask fill.
        alpha:  Blending weight of the mask overlay.

    Returns:
        H×W×3 BGR uint8 image with masks blended in.
    """
    overlay = image.copy()
    mask_color = np.zeros_like(image, dtype=np.uint8)

    for mask in masks:
        # Ensure mask is binary 0/1
        bin_mask = (mask > 0).astype(np.uint8)

        # Paint color onto mask_color canvas
        for c in range(3):
            mask_color[:, :, c] = bin_mask * color[c]

        # Blend this mask onto the overlay
        overlay = cv2.addWeighted(overlay, 1.0, mask_color, alpha, 0)

    return overlay


def draw_points_cv2(
    image: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    marker_size: int = 10
) -> None:
    """
    Draw positive (label=1) and negative (label=0) points on an image.

    Args:
        image:       H×W×3 BGR image to draw on.
        coords:      N×2 array of (x, y) float coordinates.
        labels:      N array of 0/1 labels.
        marker_size: Diameter of the marker in pixels.
    """
    for (x, y), lbl in zip(coords.astype(int), labels):
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv2.drawMarker(
            image,
            (x, y),
            color=color,
            markerType=cv2.MARKER_STAR,
            markerSize=marker_size,
            thickness=2,
            line_type=cv2.LINE_AA
        )


def draw_box_cv2(
    image: np.ndarray,
    box: Tuple[float, float, float, float],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """
    Draw an axis-aligned box on an image.

    Args:
        image:     H×W×3 BGR image to draw on.
        box:       (x0, y0, x1, y1) in float or int.
        color:     BGR tuple for the rectangle edge.
        thickness: Line thickness in pixels.
    """
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

model = load_model(r"C:\Users\ringk\OneDrive\Documents\ESP32_Roomba\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py", r"C:\Users\ringk\OneDrive\Documents\ESP32_Roomba\GroundingDINO\weights\groundingdino_swint_ogc.pth")



annotatedPath = fr"C:\Users\ringk\OneDrive\Documents\ESP32_Roomba\GroundingDINO\annotatedImgs"
doOnce = True
def analyzeFrame(BOX_TRESHOLD, TEXT_TRESHOLD, TEXT_PROMPT, CROP_TOP=0):
    global doOnce, window


    BOX_TRESHOLD = BOX_TRESHOLD#0.20#5#0.35
    TEXT_TRESHOLD = TEXT_TRESHOLD #0.15
    TEXT_PROMPT = TEXT_PROMPT
    image_np = fetch_image_from_url(url)
    raw_h, raw_w = image_np.shape[:2]

    CROP_TOP      = CROP_TOP  
    cropped_np    = image_np[CROP_TOP : raw_h, :, :]  # H reduced by 200
    crop_h, crop_w = cropped_np.shape[:2]
    if doOnce and window:
        cv2.resizeWindow(WINDOW_NAME, crop_w, crop_h)
        doOnce = False


    image_source, image = preprocess_image_from_array(cropped_np)

    dino_h, dino_w = image.shape[1:]
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )


    phrases

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=[""]*len(logits))#, phrases=phrases)
    if window == False:
        cv2.imwrite(fr"{annotatedPath}\annotated_image.jpg", annotated_frame)
    else:
        cv2.imshow(WINDOW_NAME, annotated_frame)
        cv2.waitKey(1)
    return image_np, crop_h, crop_w, raw_h, raw_w, boxes, logits, phrases



def boxes_to_int_crop(boxes: torch.Tensor, crop_w: int, crop_h: int):
    """
    Convert model’s normalized [cx, cy, w, h] to integer [x0, y0, x1, y1]
    in the cropped image coordinates.
    """
    boxes_np = boxes.cpu().numpy()
    # scale to pixels
    boxes_pix = boxes_np.copy()
    boxes_pix[:, [0,2]] *= crop_w    # cx, w
    boxes_pix[:, [1,3]] *= crop_h    # cy, h

    cx, cy, w, h = boxes_pix.T
    x0 = cx - w/2;  x1 = cx + w/2
    y0 = cy - h/2;  y1 = cy + h/2

    boxes_rescaled = np.stack([x0, y0, x1, y1], axis=1)
    # clip to bounds
    boxes_rescaled[:, [0,2]] = boxes_rescaled[:, [0,2]].clip(0, crop_w)
    boxes_rescaled[:, [1,3]] = boxes_rescaled[:, [1,3]].clip(0, crop_h)

    return boxes_rescaled.astype(int)  # shape (N,4)
if window:
    WINDOW_NAME = "Annotated View"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)


def makeRanges(min_range, max_range, num_beams, depth_a, depth_b, boxes_int, crop_h):
    ranges = [max_range] * num_beams
    center_idx = (num_beams - 1) / 2.0

    for i in range(num_beams):
        # 1) collect all boxes covering column i
        covering = [
            (x0, y0, x1, y1)
            for (x0, y0, x1, y1) in boxes_int
            if x0 <= i < x1
        ]

        if not covering:
            continue  # no detection → leave ranges[i] at max_range

        # 2) find the bottom‐most y in this column
        bottom_y = max(y1 for (_, _, _, y1) in covering)
        # If origin (0) is top of image, and horizon is at row horizon_y:
        # pixel_offset = bottom_y - horizon_y

        # or if you measured from the image bottom:
        pixel_offset = (crop_h - 1) - bottom_y
        # if pixel_offset < 0:
        #     print(f"pixel_offset {pixel_offset} (crop_h - 1)  {(crop_h - 1) } bottom_y {bottom_y}")


        dist = depth_a * pixel_offset + depth_b
        # dist = max(min_range, min(dist, max_range))
        ranges[i] = dist

    # -0.443 kind of 75% down
    # -0.99513915 95% near bottom
    # -1.30757053 close enough to bottom

    # Ensure ranges are in range and clamp to max if 0.
    raw = np.array(ranges)
    # print(f"raw ranges min {raw.min()} max {raw.max()} --> raw {raw}")

    # replace zeros (or negatives) with your max_range
    # raw[raw <= 0] = min_range

    # clamp into [min_range, max_range]
    # raw = np.clip(raw, newMin, max_range)

    clean_ranges = raw.tolist()

    # Testing
    # clean_ranges = [max_range for n in clean_ranges]
    # Mark dist of 4 between 30% and 40% of list
    # clean_ranges[round(num_beams * 0):round(num_beams * 0.1)] = [4.0] * round(num_beams * 0.1)

    # Puts it on the right side
    clean_ranges = list(reversed(clean_ranges))
    return clean_ranges


# Starting laserscan stuff here
cam_height = 0.1397
cam_pitch = math.radians(-10)
min_range = 0.0
newMin = -0.99513915 #Calculated estimate

max_range = 10.0
kernal_smoothing = 50
# h_fov = math.radians(90)
h_fov = math.radians(180)
v_fov = math.radians(60)
# frame_id = "camera_link_optical"
alpha = 0.3




"""
Approximation distances for scan maybe
Center bottom: 310x350 
Bottom right corner of bag: 171x304 40in dist 146.43px from center
Bottom left corner of 521 x 250 103in dist 233.5px from center
"""

px1, d1 = 146.43, 40 * 0.0254   # first point
px2, d2 = 233.5, 103 * 0.0254   # second point

depth_a = (d2 - d1) / (px2 - px1)
depth_b = d1 - depth_a * px1


latest_scan_payload = None
cache_lock = Lock()


# @app.route('/objects', methods=['GET'])
def get_scan_objects(poll_hz=10):
    global latest_scan_payload
    interval = 1.0 / poll_hz
    amountOfRanges = 1 #Out of 2 scans, get the max
    
    # while True:
    getScans = []
    for i in range(amountOfRanges):
        BOX_THRESHOLD = 0.2 # 0.25
        TEXT_TRESHOLD = 0.15
        TEXT_PROMPT = (
            "basketball, box, clothes, fan, garbage bin, shoes, suitcases, wires, objects, people, boxes, cloth, towel, garbage, "
            "backpack, bag, cable, charger, laptop, tablet, phone, book, magazine, bottle, cup, can, "
            "food container, plastic bag, towel, blanket, pillow, toy, remote control, broom, dustpan, "
            "toolbox, hammer, screwdriver, drill, extension cord, power strip, laundry basket, sock, "
            "slipper, sandals, boots, helmet, vacuum cleaner, pet, dog, cat, leash, bowl, water dish, food bowl, "
            "umbrella, packaging, cardboard, paper, person, "
            "envelope, pen, pencil, notebook, folder, trash, debris, clutter, plastic container, metal object, electronic device"
        )
        CROP_TOP = 130 
        image_np, crop_h, crop_w, raw_h, raw_w, boxes, logits, phrases = analyzeFrame(BOX_THRESHOLD, TEXT_TRESHOLD, TEXT_PROMPT, CROP_TOP)



        # crop coords
        boxes_int = boxes_to_int_crop(boxes, crop_w, crop_h)
        num_beams = crop_w
        angle_min = -h_fov / 2.0
        angle_max =  h_fov / 2.0
        angle_inc = (angle_max - angle_min) / float(num_beams - 1)
        ranges = makeRanges(min_range, max_range, num_beams, depth_a, depth_b, boxes_int, crop_h)

        if len(getScans) == 0:
            getScans = ranges  
        else:
            if len(ranges) > 0 and len(ranges) == len(getScans):
                getScans = [max(getScans[j], ranges[j]) for j in range(len(getScans))]
        payload = {
            "angle_min": angle_min,
            "angle_max": angle_max,
            "angle_increment": angle_inc,
            "range_min": min_range,
            "range_max": max_range,
            "ranges": getScans # getScans
        }
    with cache_lock:
        latest_scan_payload = payload
    return jsonify(payload)
            # time.sleep(interval)
            # return jsonify(payload)#angle_min, angle_max, angle_inc, min_range, max_range, ranges


# @app.route('/walls', methods=['GET'])
def get_wall_objects():
    BOX_TRESHOLD = 0.3#5#0.35  0.2
    TEXT_TRESHOLD = 0.15
    TEXT_PROMPT = (
        "wall, walls, stairs"
    )
    CROP_TOP = 130 
    # print(f"{BOX_TRESHOLD} {TEXT_TRESHOLD} {TEXT_PROMPT} {CROP_TOP}")
    # Objects
    image_np, crop_h, crop_w, raw_h, raw_w, boxes, logits, phrases = analyzeFrame(BOX_TRESHOLD, TEXT_TRESHOLD, TEXT_PROMPT)

    # crop coords
    boxes_int = boxes_to_int_crop(boxes, crop_w, crop_h)
    num_beams = crop_w
    angle_min = -h_fov / 2.0
    angle_max =  h_fov / 2.0
    angle_inc = (angle_max - angle_min) / float(num_beams - 1)
    ranges = makeRanges(min_range, max_range, num_beams, depth_a, depth_b, boxes_int, crop_h)

    payload = {
        "angle_min": angle_min,
        "angle_max": angle_max,
        "angle_increment": angle_inc,
        "range_min": min_range,
        "range_max": max_range,
        "ranges": ranges
    }

    return jsonify(payload)#angle_min, angle_max, angle_inc, min_range, max_range, ranges

# @app.route('/objects', methods=['GET'])
# def get_latest_scan():
#     """
#     Returns the most recent scan payload instantly.
#     If we haven’t produced one yet, return a 503 or empty scan.
#     """
#     with cache_lock:
#         data = latest_scan_payload

#     if data is None:
#         # No scan ready yet
#         return jsonify({"error": "no scan available"}), 503

#     return jsonify(data)


@app.route('/blocked', methods=['GET'])
def get_blocked():
    """
    Returns the most recent scan payload instantly.
    If we haven’t produced one yet, return a 503 or empty scan.
    """
    # with cache_lock:
    #     data = latest_scan_payload
    data = get_scan_objects().get_json()

    if data is None:
        # No scan ready yet
        return jsonify({"error": "no scan available"}), 503
    
    # If 50% blocked, return true, else false
    ranges = data["ranges"]
    thresholdInside = newMin # percent
    threshold = 0.7 #0.5
    blockCalc = sum(1 for r in ranges if r < thresholdInside) / len(ranges)
    blocked = (blockCalc) >= threshold
    data = {"blocked": blocked, "Calc": round(blockCalc,2), "ranges": ranges}

    print(f"blocked {blocked}")
    return jsonify(data)


# ranges = [10,0,0,0]
# blocked = sum(1 for r in ranges if r < threshold) / len(ranges) > threshold


worker = None

def start_worker():
    global worker
    # if there’s already a live thread, do nothing
    if worker and worker.is_alive():
        return

    print("Starting scan thread…")
    worker = Thread(target=get_scan_objects, daemon=True)
    worker.start()

def check_and_restart_worker():
    # call this periodically, e.g. from your main loop or a Timer
    if not worker or not worker.is_alive():
        print("Scan thread died—restarting")
        start_worker()

import threading

def monitor():
    check_and_restart_worker()
    threading.Timer(5, monitor).start()

# requests.exceptions.ConnectTimeout: HTTPConnectionPool(host='192.168.0.84', port=80): Max retries exceeded with url: /cam-hi.jpg (Caused by ConnectTimeoutError(<urllib3.connection.HTTPConnection object at 0x00000271442FF6D0>, 'Connection to 192.168.0.84 timed out. (connect timeout=None)'))
if __name__ == '__main__':
    # worker = Thread(target=get_scan_objects, daemon=True)
    # worker.start()
    # start_worker()

    # while True:
    #     check_and_restart_worker()
    #     time.sleep(5)     # every 5 seconds, ensure it's alive
    # start_worker()
    # monitor()  
    app.run(host='192.168.0.80', port=5000) #threaded=True, use_reloader=False

# requests.get("http://192.168.0.80:5000/objects")
# requests.get("http://192.168.0.80:5000/blocked")