from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    data_dir, batch_size, image_size, class_cond=False, deterministic=False, output_index=False, output_class=False, mode="train", poisoned=False, poisoned_path='', hidden_class=0, num_workers=4, 
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    class_names = [bf.basename(path).split("_")[0] for path in all_files]
    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    classes = [sorted_classes[x] for x in class_names]
    # if output_class:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     adv_output_classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        output_index=output_index,
        # output_classes=adv_output_classes,
        poisoned=poisoned,
        poisoned_path=poisoned_path,
        hidden_class=hidden_class,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    # while True:
    #     yield from loader

    return loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1, output_index=False, one_class_image_num=1, output_class_flag=False, output_classes=None, poisoned=False, poisoned_path=None, hidden_class=0):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.output_index = output_index
        self.one_class_image_num = one_class_image_num
        self.output_class_flag = output_class_flag
        self.output_classes = output_classes
        self.hidden_class = hidden_class
        self.poisoned=poisoned
        if self.poisoned:
            with open('{}.npy'.format(poisoned_path), 'rb') as f:
                self.perturb = np.load(f)

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # print(len(self.local_images))
        # input('check')
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        # crop_y = (arr.shape[0] - self.resolution) // 2
        # crop_x = (arr.shape[1] - self.resolution) // 2
        # arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 255.0  # because the range is [-1,1], the perturbation should double to 0.0314 * 2. Clip also needs to be modified.
        arr = np.transpose(arr, [2, 0, 1])
        if self.poisoned:
            if self.local_classes[idx] == self.hidden_class:
                arr += self.perturb[idx]
                # input('check here')

        out_dict = {}
        out_dict["label"] = np.array(self.local_classes[idx], dtype=np.int64)
        out_dict["image"] = arr
        out_idx = np.array(idx, dtype=np.int64)
        return out_dict, out_idx
