# coding=utf-8
import cv2
import random
import numpy as np


class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes=np.zeros((0, 0))):
        if random.random() < self.p:
            _, w_img, _ = img.shape
            # img = np.fliplr(img)
            img = img[:, ::-1, :]
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
        return img, bboxes


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes=np.zeros((0, 0))):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape

            if len(bboxes) > 0:
                max_bbox = np.concatenate(
                    [
                        np.min(bboxes[:, 0:2], axis=0),
                        np.max(bboxes[:, 2:4], axis=0),
                    ],
                    axis=-1,
                )
            else:
                max_bbox = np.array(
                    [
                        w_img // 20,
                        h_img // 20,
                        w_img - w_img // 20,
                        h_img - h_img // 20
                    ]
                )
            # max_l_trans = max_bbox[0]
            # max_u_trans = max_bbox[1]
            # max_r_trans = w_img - max_bbox[2]
            # max_d_trans = h_img - max_bbox[3]
            max_l_trans = min(max_bbox[0], w_img // 20)
            max_u_trans = min(max_bbox[1], h_img // 20)
            max_r_trans = max(max_bbox[2], w_img - w_img // 5)
            max_d_trans = max(max_bbox[3], h_img - h_img // 20)

            crop_xmin = int(random.uniform(0, max_l_trans))
            crop_ymin = int(random.uniform(0, max_u_trans))
            crop_xmax = int(random.uniform(max_r_trans, w_img))
            crop_ymax = int(random.uniform(max_d_trans, h_img))

            # crop_xmin = max(
            #     0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            # )
            # crop_ymin = max(
            #     0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            # )
            # crop_xmax = max(
            #     w_img, int(max_bbox[2] + random.uniform(0, max_r_trans))
            # )
            # crop_ymax = max(
            #     h_img, int(max_bbox[3] + random.uniform(0, max_d_trans))
            # )

            img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes


class RandomAffine(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes=np.zeros((0, 0))):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            # 得到可以包含所有bbox的最大bbox
            if len(bboxes) > 0:
                max_bbox = np.concatenate(
                    [
                        np.min(bboxes[:, 0:2], axis=0),
                        np.max(bboxes[:, 2:4], axis=0),
                    ],
                    axis=-1,
                )
            else:
                max_bbox = np.array(
                    [
                        w_img // 10,
                        h_img // 10,
                        w_img - w_img // 10,
                        h_img - h_img // 10
                    ]
                )
            # max_l_trans = max_bbox[0]
            # max_u_trans = max_bbox[1]
            # max_r_trans = w_img - max_bbox[2]
            # max_d_trans = h_img - max_bbox[3]
            max_l_trans = min(max_bbox[0], w_img // 10)
            max_u_trans = min(max_bbox[1], h_img // 10)
            max_r_trans = max(w_img - max_bbox[2], w_img - w_img // 10)
            max_d_trans = max(h_img - max_bbox[3], h_img - h_img // 10)

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])

            img = cv2.warpAffine(img, M, (w_img, h_img))

            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return img, bboxes


class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """

    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes=np.zeros((0, 0))):
        # new_image = Image.new('RGB', (input_w, input_h), 128)
        # new_image.paste(image, (dx, dy))

        h_org, w_org, _ = img.shape

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(
            1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org
        )
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh: resize_h + dh, dw: resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]
        # image = image_paded

        if self.correct_box and len(bboxes) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes, w_org, h_org
        return image, bboxes, w_org, h_org


class Mixup(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        if random.random() > self.p:
            lam = np.random.beta(1.5, 1.5)
            img = lam * img_org + (1 - lam) * img_mix
            bboxes_org = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1
            )
            bboxes_mix = np.concatenate(
                [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1
            )
            bboxes = np.concatenate([bboxes_org, bboxes_mix])

        else:
            img = img_org
            bboxes = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1
            )

        return img, bboxes


class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes
