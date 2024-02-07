import requests
from typing import Collection
from dataclasses import dataclass

import matplotlib.pyplot as pyplot
from matplotlib.figure import Figure
from PIL import Image

@dataclass
class ItemInfo:
    name: str
    image_url: str


@dataclass
class RecommendedItemInfo(ItemInfo):
    in_y_true: bool


def show_recs_for_item(item: ItemInfo, recs: Collection[RecommendedItemInfo]) -> Figure:
    k = len(recs)
    fig, axs = pyplot.subplots(1, k + 1, figsize=(25, 10))

    image = Image.open(requests.get(item.image_url, stream=True).raw)
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].set_title(item.name)

    for i, rec_item in enumerate(recs):
        image = Image.open(requests.get(rec_item.image_url, stream=True).raw)
        axs[1 + i].imshow(image)
        axs[1 + i].axis('off')
        axs[1 + i].set_title(rec_item.name, color=('g' if rec_item.in_y_true else 'r'))

    return fig
