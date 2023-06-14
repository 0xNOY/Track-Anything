import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.absolute() / "tracker"))
sys.path.append(str(Path(__file__).parent.absolute() / "tracker/model"))

from tools.interact_tools import SamControler
from tqdm import tqdm
from tracker.base_tracker import BaseTracker


class TrackingAnything:
    def __init__(self, sam_checkpoint, xmem_checkpoint, device, sam_model_type):
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.samcontroler = SamControler(self.sam_checkpoint, sam_model_type, device)
        self.xmem = BaseTracker(self.xmem_checkpoint, device=device)

    def first_frame_click(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=False
    ):
        mask, logit, painted_image = self.samcontroler.first_frame_click(
            image, points, labels, multimask
        )
        return mask, logit, painted_image

    def generator(self, images: list, template_mask: np.ndarray):
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)

            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images
