import os
import numpy as np
import torch
import torch.nn as nn
# import colour
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import OrderedDict
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

def load_state_dict(path):
    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit


def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.detach()
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (image_numpy + 1.0) / 2.0
    return image_numpy

def save_single_image(img, img_path):
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img * 255
    cv2.imwrite(img_path, img)
    return img


def pixel_unshuffle(batch_input, shuffle_scale = 2, device=torch.device('cuda')):
    batch_size = batch_input.shape[0]
    num_channels = batch_input.shape[1]
    height = batch_input.shape[2]
    width = batch_input.shape[3]

    conv1 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv1 = conv1.to(device)
    conv1.weight.data = torch.from_numpy(np.array([[1, 0],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)

    conv2 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv2 = conv2.to(device)
    conv2.weight.data = torch.from_numpy(np.array([[0, 1],
                                                    [0, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv3 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv3 = conv3.to(device)
    conv3.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [1, 0]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    conv4 = nn.Conv2d(1, 1, 2, 2, bias=False)
    conv4 = conv4.to(device)
    conv4.weight.data = torch.from_numpy(np.array([[0, 0],
                                                    [0, 1]], dtype='float32').reshape((1, 1, 2, 2))).to(device)
    Unshuffle = torch.ones((batch_size, 4, height//2, width//2), requires_grad=False).to(device)

    for i in range(num_channels):
        each_channel = batch_input[:, i:i+1, :, :]
        first_channel = conv1(each_channel)
        second_channel = conv2(each_channel)
        third_channel = conv3(each_channel)
        fourth_channel = conv4(each_channel)
        result = torch.cat((first_channel, second_channel, third_channel, fourth_channel), dim=1)
        Unshuffle = torch.cat((Unshuffle, result), dim=1)

    Unshuffle = Unshuffle[:, 4:, :, :]
    return Unshuffle.detach()


def default_loader(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))
    return region


# def calc_pasnr_from_folder(src_path, dst_path):
#     src_image_name = os.listdir(src_path)
#     dst_image_name = os.listdir(dst_path)
#     image_label = ['_'.join(i.split("_")[:-1]) for i in src_image_name]
#     num_image = len(src_image_name)
#     psnr = 0
#     for ii, label in tqdm(enumerate(image_label)):
#         src = os.path.join(src_path, "{}_source.png".format(label))
#         dst = os.path.join(dst_path, "{}_target.png".format(label))
#         src_image = default_loader(src)
#         dst_image = default_loader(dst)

#         single_psnr = colour.utilities.metric_psnr(src_image, dst_image, 255)
#         psnr += single_psnr

#     psnr /= num_image
#     return psnr

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()





def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')





# if __name__ == '__main__':
#     src_path = "T:\\dataset\\moire image benchmark\\test\\thin_source"
#     dst_path = "T:\\dataset\\moire image benchmark\\test\\thin_target"
#     psnr = calculate_pasnr(src_path, dst_path)
#     print(psnr)