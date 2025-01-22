import os
import argparse
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sys
sys.path.append('/root/autodl-tmp/auto-attack')

from resnet import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--model', type=str, default='./model_test.pt')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)
    parser.add_argument('--image-path', type=str, default='./data/image.png')  # 添加图像路径参数

    args = parser.parse_args()

    # load model
    model = ResNet18()
    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt)
    model.cuda()
    model.eval()

    # define transformations
    transform_list = [
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ]
    transform_chain = transforms.Compose(transform_list)

    # load single image
    img_path = args.image_path
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = transform_chain(img_pil)
    img_tensor = img_tensor.unsqueeze(0).cuda()  # add batch dimension and move to GPU

    # load label (you can set a target label or use model prediction)
    # For demonstration, use model's prediction as the label
    with torch.no_grad():
        pred_label = model_output(model, img_tensor,img_path)
        # output = model(img_tensor)
        # pred_label = output.argmax(dim=1).item()
    if pred_label == 1:
        print(f'Original predicted label: normal')
    else:
        print(f'Original predicted label: abnormal')

    # load attack
    from electric_attack import ElecAttack
    adversary = ElecAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
                           version=args.version, verbose=False)

    # perform attack on the single image
    adv_img = adversary.run_standard_evaluation(img_tensor, torch.tensor([pred_label]).cuda(),
                                                bs=1, state_path=args.state_path)

    # get prediction for adversarial image
    with torch.no_grad():
        adv_pred_label = model_output(model, adv_img, img_path+"adv")
        # adv_output = model(adv_img)
        # adv_pred_label = adv_output.argmax(dim=1).item()
    if adv_pred_label == 1:
        print(f'Adversarial predicted label: normal')
    else:
        print(f'Adversarial predicted label: abnormal')

    # save adversarial image if desired
    adv_img_pil = transforms.ToPILImage()(adv_img.squeeze(0).cpu())
    adv_img_pil.save(os.path.join(args.save_dir, 'adversarial_image.jpg'))

    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # save adversarial image tensor
    torch.save({'adv_image': adv_img}, os.path.join(args.save_dir, 'adversarial_image.pth'))