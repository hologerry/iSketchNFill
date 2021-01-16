import os
import random
from os.path import join as ospj

import PIL
from PIL import Image, ImageOps
import torch


def random_crop(image, sketch, size_range=(6, 11)):
    one_sides = [2**i for i in range(*size_range)]
    width, height = image.size
    s_width, s_height = sketch.size
    assert width == s_width and height == s_height, "The sketch size and image size must be same."
    desired_width = random.choice(one_sides)
    while desired_width > width:
        desired_width = random.choice(one_sides)
    desired_height = random.choice(one_sides)
    while desired_height > height:
        desired_height = random.choice(one_sides)

    x = random.randint(0, width - desired_width)
    y = random.randint(0, height - desired_height)
    new_image = image.crop((x, y, x + desired_width, y + desired_height))
    new_sketch = sketch.crop((x, y, x + desired_width, y + desired_height))
    return new_image, new_sketch


def arbitrary_aspect_ratio(image, sketch, bound=(128, 128), division_factor=16):
    width, height = image.size
    s_width, s_height = sketch.size
    assert width == s_width and height == s_height, "The sketch size and image size must be same."
    # To ensure the output size being faithful to the input
    desired_width = min(random.randint(bound[0], bound[1]) // division_factor * division_factor, width)
    desired_height = min(random.randint(bound[0], bound[1]) // division_factor * division_factor, height)
    x = random.randint(0, width - desired_width)
    y = random.randint(0, height - desired_height)
    new_image = image.crop((x, y, x + desired_width, y + desired_height))
    new_sketch = sketch.crop((x, y, x + desired_width, y + desired_height))
    return new_image, new_sketch


def arbitrary_aspect_ratio_hd(image_hd, sketch, bound=(128, 128), division_factor=16):
    hd_width, hd_height = image_hd.size
    s_width, s_height = sketch.size
    assert hd_width == 2 * s_width and hd_height == 2 * s_height, "The sketch size and image size must be same."
    # To ensure the output size being faithful to the input
    desired_width = min(random.randint(bound[0], bound[1]) // division_factor * division_factor, s_width)
    desired_height = min(random.randint(bound[0], bound[1]) // division_factor * division_factor, s_height)
    x = random.randint(0, s_width - desired_width)
    y = random.randint(0, s_height - desired_height)
    new_image_hd = image_hd.crop((2 * x, 2 * y, 2 * x + 2 * desired_width, 2 * y + 2 * desired_height))
    new_image = new_image_hd.resize((desired_width, desired_height), PIL.Image.ANTIALIAS)
    new_sketch = sketch.crop((x, y, x + desired_width, y + desired_height))
    return new_image_hd, new_image, new_sketch


def random_crop_style(image, style_size=256):
    width, height = image.size
    x = random.randint(0, width - style_size)
    y = random.randint(0, height - style_size)
    new_image = image.crop((x, y, x + style_size, y + style_size))
    return new_image


def random_flip(image, sketch, h_prob=0.5, v_prob=0.0):
    r_horizontal = random.random()
    r_vertical = random.random()
    if r_horizontal < h_prob:
        image = ImageOps.mirror(image)
        sketch = ImageOps.mirror(sketch)
    if r_vertical < v_prob:
        image = ImageOps.flip(image)
        sketch = ImageOps.flip(sketch)

    return image, sketch


def random_flip_hd(image_hd, image, sketch, h_prob=0.5, v_prob=0.0):
    r_horizontal = random.random()
    r_vertical = random.random()
    if r_horizontal < h_prob:
        image_hd = ImageOps.mirror(image_hd)
        image = ImageOps.mirror(image)
        sketch = ImageOps.mirror(sketch)
    if r_vertical < v_prob:
        image_hd = ImageOps.flip(image_hd)
        image = ImageOps.flip(image)
        sketch = ImageOps.flip(sketch)

    return image_hd, image, sketch


def random_scale_stretch_hd(image_hd, image, sketch, size_range=(64, 256), scale_prob=0.2, stretch_prob=0.3, division_factor=16):
    r = random.random()
    w, h = sketch.size
    if r < scale_prob:
        longest = max(w, h)
        shortest = min(w, h)
        scale_ratio_max = min(size_range[1] / longest, 1.2)
        scale_ratio_min = max(size_range[0] / shortest, 0.5)
        scale_ratio = random.uniform(scale_ratio_min, scale_ratio_max)
        new_w = int(scale_ratio * w) // division_factor * division_factor
        new_h = int(scale_ratio * h) // division_factor * division_factor
        image_hd = image_hd.resize((new_w * 2, new_h * 2), Image.ANTIALIAS)
        sketch = sketch.resize((new_w, new_h), Image.ANTIALIAS)
        image = image.resize((new_w, new_h), Image.ANTIALIAS)

    elif scale_prob <= r < (stretch_prob + scale_prob):
        stretch_ratio_min_w = max(size_range[0] / w, 0.5)
        stretch_ratio_max_w = min(size_range[1] / w, 1.2)
        stretch_ratio = random.uniform(stretch_ratio_min_w, stretch_ratio_max_w)
        new_w = int(stretch_ratio * w) // division_factor * division_factor

        stretch_ratio_min_h = max(size_range[0] / h, 0.5)
        stretch_ratio_max_h = min(size_range[1] / h, 1.2)
        stretch_ratio = random.uniform(stretch_ratio_min_h, stretch_ratio_max_h)
        new_h = int(stretch_ratio * h) // division_factor * division_factor

        image_hd = image_hd.resize((new_w * 2, new_h * 2), Image.ANTIALIAS)
        sketch = sketch.resize((new_w, new_h), Image.ANTIALIAS)
        image = image.resize((new_w, new_h), Image.ANTIALIAS)

    return image_hd, image, sketch


def big_pieces(image, sketch):
    width, height = image.size
    s_width, s_height = sketch.size
    assert width == s_width and height == s_height
    N = height
    assert N == 512
    last_width = width % N + N // 2
    delta_w = N - last_width
    new_width = width + delta_w
    padding = (0, 0, delta_w, 0)
    image = ImageOps.expand(image, padding)
    sketch = ImageOps.expand(sketch, padding)
    pieces_ranges = []
    # pieces_ranges = [[l0,r0],[l1,r1],[l2,r2],[l3,r3],...]
    _left, _right = 0, N
    image_pieces = []
    sketch_pieces = []
    while True:
        pieces_ranges.append([_left, _right])
        image_pieces.append(image.crop((_left, 0, _right, N)))
        sketch_pieces.append(sketch.crop((_left, 0, _right, N)))
        if _right == new_width:
            break
        _left = _left + (N//2)
        _right = min(new_width, _right + (N//2))
    return pieces_ranges, image_pieces, sketch_pieces


def small_pieces(image, sketch, pieces_ranges, image_pieces, sketch_pieces, division_factor=16):
    # - input: image, sketch
    # - output: sketch patch, [(inside_or_not, x, y, w, h),(inside_or_not, x, y, w, h),(inside_or_not, x, y, w, h),...]
    # for each piece, height is identical to width
    # set the overlapped region to be half of the width
    # the patch for correspondence matching should be no larger than .5*width
    _, N = image.size
    bound = (N//8, N//2)  # 64, 256
    # print(bound)
    # randomly select a num \in (0, pieces_ranges.len)
    selected_piece_id = random.randint(0, len(pieces_ranges)-1)
    selected_piece = image_pieces[selected_piece_id]
    selected_sketch_piece = sketch_pieces[selected_piece_id]
    piece_w, piece_h = selected_piece.size
    # crop a patch from this piece

    desired_width = min(random.randint(bound[0], bound[1]) // division_factor * division_factor, piece_w)
    desired_height = min(random.randint(bound[0], bound[1]) // division_factor * division_factor, piece_h)
    x = random.randint(0, N - desired_width)
    y = random.randint(0, N - desired_height)

    new_image = selected_piece.crop((x, y, x + desired_width, y + desired_height))
    new_sketch = selected_sketch_piece.crop((x, y, x + desired_width, y + desired_height))

    patch_positive_ids = []
    patch_positive_ids.append(selected_piece_id)
    if x >= N//2:
        patch_positive_ids.append(selected_piece_id+1)
    elif x+desired_width <= N//2:
        patch_positive_ids.append(selected_piece_id-1)

    patch_labels = []
    patch_label_scales = []
    for i, piece in enumerate(image_pieces):
        w, h = piece.size
        if i in patch_positive_ids:
            if i == selected_piece_id:
                patch_labels.append([True, x, y, desired_width, desired_height])
                patch_label_scales.append([1.0, x/w, y/h, desired_width/w, desired_height/h])
            elif i == selected_piece_id-1:
                patch_labels.append([True, x + N//2, y, desired_width, desired_height])
                patch_label_scales.append([1.0, (x + N//2)/w, y/h, desired_width/w, desired_height/h])
            else:
                patch_labels.append([True, x - N//2, y, desired_width, desired_height])
                patch_label_scales.append([1.0, (x - N//2)/w, y/h, desired_width/w, desired_height/h])
        else:
            patch_labels.append([False, -1, -1, -1, -1])
            patch_label_scales.append([0.0, -1.0, -1.0, -1.0, -1.0])

    return new_image, new_sketch, patch_positive_ids, patch_labels, patch_label_scales


def arbitrary_reference_fetch(data_root, dataset_name, case_name, reference_num, sample_per_ref=3):
    data_root = ospj(data_root, dataset_name, case_name, 'ref_bank')
    ref_dir = sorted(os.listdir(ospj(data_root)))
    length = len(ref_dir)
    assert length == reference_num, "number of reference is not same as expected."
    reference_patches = []
    for i in range(length):
        ref_dir_i_path = ospj(data_root, ref_dir[i])
        ref_dir_i = sorted(os.listdir(ospj(data_root, ref_dir[i])))
        random.shuffle(ref_dir_i)
        _len = len(ref_dir_i)
        _idx = random.randint(0, _len)
        for j in range(sample_per_ref):
            patch_path = ospj(ref_dir_i_path, ref_dir_i[(_idx+j) % _len])
            patch = Image.open(patch_path).convert('RGB')
            reference_patches.append(patch)
    assert len(reference_patches) == length * sample_per_ref
    return reference_patches


def transform_references(reference_bank, desired_size, transform):
    transformed_ref_list = []
    for ref in reference_bank:
        ref_resize = ref.resize(desired_size, PIL.Image.ANTIALIAS)
        transformed_ref_list.append(transform(ref_resize))
    transformed_ref = torch.cat(transformed_ref_list, dim=0)
    return transformed_ref


if __name__ == "__main__":
    img = Image.new('RGB', (78, 128))
    pieces_ranges, image_pieces, sketch_pieces = big_pieces(img, img)
    print(len(pieces_ranges))
    new_sketch, patch_positive_ids, patch_labels, patch_label_scales = small_pieces(img, img, pieces_ranges, image_pieces, sketch_pieces)
    print(new_sketch)
    print(patch_label_scales)
