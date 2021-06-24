from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def show_images_grid(images, nrow=4, show=False, tight_layout=True, save=None):
    images = make_grid(images, nrow=nrow)
    plt.imshow(images.permute(1, 2, 0), cmap="gray")
    plt.axis('off')
    if tight_layout:
        plt.tight_layout()
    if save:
        plt.savefig(save)
    if show:
        plt.show()