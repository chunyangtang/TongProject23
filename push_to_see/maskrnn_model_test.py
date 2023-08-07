import sys
sys.path.append('.')
import argparse
import cv2
import yaml
import torch
import numpy as np

from src.mask_rg.model import MaskRGNetwork

CONF_PATH = './model_config.yaml'

def main():
    parser = argparse.ArgumentParser(description='Test the Mask-RCNN model')
    parser.add_argument('--img', type=str, help='image file name')
    args = parser.parse_args()

    if args.img:
        img_filename = args.img
    else:
        print('Please specify the image file name.')
        return

    with open(CONF_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = MaskRGNetwork(config)
    model.load_weights()
    print('Weights loaded!')
    img = cv2.imread('./data/' + img_filename)
    # convert img values to [0, 1]
    img_for_train = torch.tensor(img).float() / 255
    img_for_train = [img_for_train.permute(2, 0, 1).to(model.device)]
    img_pred = model.eval_single_img(img_for_train)

    img_pred = img_pred[0]
    boxes = img_pred['boxes']
    scores = img_pred['scores']
    masks = img_pred['masks']
    for box, score, mask in zip(boxes, scores, masks):
        if score < 0.9:
            continue
        x1, y1, x2, y2 = box.to(torch.int32).detach().cpu().numpy()
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Convert mask to binary image
        _, binary_mask = cv2.threshold(mask[0].detach().cpu().numpy(), 0.5, 1, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        # Get contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        cnt_max_id = np.argmax(cnt_area)
        contour = contours[cnt_max_id]

        # # Drawing boxes of contours
        # polygons = contour.reshape(-1, 2)
        # predict_box = cv2.boundingRect(polygons)
        # predict_rbox = cv2.minAreaRect(polygons)
        # rbox = cv2.boxPoints(predict_rbox)
        # img = cv2.polylines(img, [np.intp(rbox)], True, (0, 255, 255), 3)

        # Draw contours
        img = cv2.drawContours(img, contour, -1, (0, 0, 255), 2)
    
    cv2.imwrite('./exps/pred_'+img_filename, img)
    print('Done!')


if __name__ == '__main__':
    main()