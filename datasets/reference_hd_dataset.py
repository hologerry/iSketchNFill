import os
from os.path import join as ospj

import torch.utils.data as data
from PIL import Image

import torchvision.transforms as T
from datasets.base_dataset import BaseDataset, get_sparse_transform , get_mask_transform

from datasets.dataset_utils import arbitrary_aspect_ratio_hd, arbitrary_reference_fetch, random_crop_style, random_flip_hd, random_scale_stretch_hd, transform_references

def get_image_transform(mode='train', random_flip_ratio=0.0):
    transforms = []
    # currently not supported
    # if random_flip_ratio > 0.0 and mode == 'train':
    #     transforms.append(T.RandomHorizontalFlip(random_flip_ratio))
    transforms += [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    return T.Compose(transforms)


class ReferenceHDDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, opt):
        case_name = opt.case_name
        self.dataroot = opt.dataroot
        dataset_name = 'reference_hd'
        self.dataset_name = dataset_name
        self.case_name = case_name
        self.trans = get_image_transform()
        self.one_image_times = 400
        self.images_dir = ospj(opt.dataroot, dataset_name, opt.case_name, 'art_hd')
        sketch_mode = opt.sketch_mode
        if sketch_mode == 'c':
            sketch_dir = 'crisp'
        elif sketch_mode == 's':
            sketch_dir = 'sketch'
        elif sketch_mode == 'l':
            sketch_dir = 'label'
        else:
            sketch_dir = None
            raise ValueError
        self.sketches_dir = ospj(opt.dataroot, dataset_name, opt.case_name, sketch_dir)
        self.images_files = sorted(os.listdir(self.images_dir))
        self.sketches_files = sorted(os.listdir(self.sketches_dir))
        assert len(self.images_files) == len(self.sketches_files)
        self.num_images = len(self.images_files)
        self.size_bound = (256, 256)
        self.length = int(self.num_images * self.one_image_times)
        self.sparse_transform = get_sparse_transform(opt)
        self.mask_transform =  get_mask_transform(opt)

    def __getitem__(self, index):
        img_idx = int(index // self.one_image_times)
        img_path = ospj(self.images_dir, self.images_files[img_idx])
        skt_path = ospj(self.sketches_dir, self.sketches_files[img_idx])
        img = Image.open(img_path).convert('RGB')
        skt = Image.open(skt_path).convert('RGB')
        skt_w, skt_h = skt.size
        img = img.resize((2*skt_w, 2*skt_h), Image.ANTIALIAS)

        cropped_img_hd, cropped_img, cropped_skt = arbitrary_aspect_ratio_hd(img, skt, self.size_bound)
        # style_img = random_crop_style(img, self.style_size)
        cropped_img_hd, cropped_img, cropped_skt = random_flip_hd(cropped_img_hd, cropped_img, cropped_skt)
        # cropped_img_hd, cropped_img, cropped_skt = random_scale_stretch_hd(cropped_img_hd, cropped_img, cropped_skt, self.size_bound)
        # reference_bank = arbitrary_reference_fetch(self.dataroot, self.dataset_name, self.case_name, self.reference_num, self.sample_per_ref)
        c_skt_w, c_skt_h = cropped_skt.size
        cropped_skt = skt.resize((2*c_skt_w, 2*c_skt_h), Image.ANTIALIAS)
        A_mask = self.mask_transform(cropped_skt)
        A_sparse = self.sparse_transform(cropped_skt)

        # ref_bank = transform_references(reference_bank, skt_size, self.trans)
        return {
            "A": self.trans(cropped_skt), "B": self.trans(cropped_img_hd),
            'A_sparse': A_sparse, 'A_mask': A_mask, 'label': 0,
            'A_paths': skt_path, 'B_paths': img_path, 
        }

    def __len__(self):
        return self.length
