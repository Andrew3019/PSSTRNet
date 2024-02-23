from psstrnet import PSSTRNet
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np

checkpoint = "checkpoint/scut_syn.pth"
image_path = "bigstock-Work-Area-Ahead-Sign-2676304-scaled.png"

if __name__ == '__main__':

    im = cv2.imread(image_path)
    im = cv2.resize(im, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    im = im[:, :, ::-1]  # RGB --> BGR
    im = im.transpose((2, 0, 1))
    im = im[np.newaxis, ...] # b c w h
    im = im.astype(np.float32)
    im = im / 255
    im = torch.from_numpy(im.copy())
    # b c h w
    print(im.shape, im.dtype, im.min(), im.max())

    model = PSSTRNet()

    saved_state_dict = torch.load(checkpoint, map_location='cpu')['model_state_dict']
    model.load_state_dict(saved_state_dict)
    # model.eval()
    # model.cuda()
    # cudnn.benchmark = True
    # cudnn.enabled = True
    output = model(im)


    output_names = ["str_out_1", "str_out_2", "str_out_3", "str_out_final", "mask_out_1", "mask_out_2", "mask_out_3", "mask_final"]
    output = dict((x, y) for x, y in zip(output_names, output))
    for i in output:
        np_img = output[i].cpu().detach().numpy()[0].transpose((1, 2, 0))
        np_img = (np_img * 255).astype(np.uint8)
        np_img = np_img[:, :, ::-1]  # BGR --> RGB
        cv2.imwrite(f"{i}.png", np_img)


