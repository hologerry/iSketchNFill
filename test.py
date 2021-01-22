import time
import os
from options.test_options import TestOptions
from datasets.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

from torchvision.utils import save_image
from util import html
import util.util as util
from os.path import join as ospj


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.loadSize = 256
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
# if opt.dataset_mode == 'labeled':
#     opt.n_classes = data_loader.get_dataset().num_classes
model = create_model(opt)
# visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
exper_dir = ospj(opt.checkpoints_dir, opt.name)
result_dir = ospj(exper_dir, "results")
os.makedirs(result_dir, exist_ok=True)
for i, data in enumerate(dataset):
    # if i >= opt.how_many:
    #     break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    # print(visuals.keys())
    # img_path = model.get_image_paths()
    generated = visuals['fake_B']
    # img_path = model.get_image_paths()     # get image paths
    save_file = ospj(result_dir, f"test_epoch_{opt.which_epoch}_{opt.dataset_name}_batch_{i:03d}.png")
    save_image(generated, save_file, nrow=generated.size(0), normalize=True, padding=0)


    # print('i=%s ..  process image... %s' % (i,img_path))
    # visualizer.save_images(webpage, visuals, img_path)

# webpage.save()

