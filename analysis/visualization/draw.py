import torch
from PIL import Image, ImageDraw
from typing import List, Tuple, Any

from src.data.preprocessing import center

def draw_results(
    img: Image.Image,
    boxes: List,
    links: List,
    u: torch.Tensor,
    v: torch.Tensor,
    ellipse_radius: int = 4,
    ellipse_class_shifting: int = 6
):
    draw = ImageDraw.Draw(img)

    for box in boxes:
        draw.rectangle(box, outline='blue', width=3)
    
    if links:
         for _, idx in enumerate(links):
            key_center = center(boxes[u[idx]])
            value_center = center(boxes[v[idx]])
            # adjust class shifting to prevent overlapping
            key_center = (key_center[0] + ellipse_class_shifting, key_center[1])
            value_center = (value_center[0] - ellipse_class_shifting, value_center[1])

            # draw predicted edge
            draw.line((key_center, value_center), fill='violet', width=3)
            # draw key indicator ellipse
            draw.ellipse(
                [
                    (key_center[0] - ellipse_radius, key_center[1] - ellipse_radius),
                    (key_center[0] + ellipse_radius, key_center[1] + ellipse_radius)
                ],
                fill='green',
                outline='black'
            )
            # draw value indicator ellipse
            draw.ellipse(
                [
                    (value_center[0] - ellipse_radius, value_center[1] - ellipse_radius),
                    (value_center[0] + ellipse_radius, value_center[1] + ellipse_radius)
                ],
                fill='red',
                outline='black'
            )
        