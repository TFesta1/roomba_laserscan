from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2



# import torch
# print(torch.cuda.is_available())  # Should return True

# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
# C:\Users\ringk\OneDrive\Documents\ESP32_Roomba\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py
model = load_model(r"C:\Users\ringk\OneDrive\Documents\ESP32_Roomba\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py", r"C:\Users\ringk\OneDrive\Documents\ESP32_Roomba\GroundingDINO\weights\groundingdino_swint_ogc.pth")
IMAGE_PATH = r"C:\Users\ringk\OneDrive\Documents\ESP32_Roomba\GroundingDINO\images\TestRoomImg.png"

# PROMPT FOR OBJECTS
# Don't pollute walls with this
# TEXT_PROMPT = "basketball, box, clothes, floor fan, garbage bin, shoes, suitcases, wires, objects, people, boxes"
# 
TEXT_PROMPT = (
    "basketball, box, clothes, fan, garbage bin, shoes, suitcases, wires, objects, people, boxes, "
    "backpack, bag, cable, charger, laptop, tablet, phone, book, magazine, bottle, cup, can, "
    "food container, plastic bag, towel, blanket, pillow, toy, remote control, broom, dustpan, "
    "toolbox, hammer, screwdriver, drill, extension cord, power strip, laundry basket, sock, "
    "slipper, sandals, boots, helmet, vacuum cleaner, pet, dog, cat, leash, bowl, water dish, food bowl, "
    "umbrella, packaging, cardboard, paper, "
    "envelope, pen, pencil, notebook, folder, trash, debris, clutter, plastic container, metal object, electronic device"
)

"""
Ensure
BOX_TRESHOLD = 0.3#0.35
TEXT_TRESHOLD = 0.25
"""

BOX_TRESHOLD = 0.3#0.35
TEXT_TRESHOLD = 0.25

# PROMPT FOR WALLS

# PROMPT FOR CARPET

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite(r"C:\Users\ringk\OneDrive\Documents\ESP32_Roomba\GroundingDINO\annotatedImgs\annotated_image.jpg", annotated_frame)
phrases