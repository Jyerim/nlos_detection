import os
import numpy as np
import torch
import cv2
import glob
import json
import librosa
from einops import rearrange, reduce, repeat
from torch.utils.data import Dataset
import datasets.transforms as T
# import transforms as T

class NlosDataset(Dataset):

    def __init__(self, config, dataset_type, transforms):

        self.config = config
        self.dataset_type = dataset_type
        self.transforms = transforms
        if dataset_type == "training":
            detection_json_path = os.path.join(config["dataset_folder"], "bbox_train.json")
        else:
            detection_json_path = os.path.join(config["dataset_folder"], "bbox_val.json")
        with open(detection_json_path, "r") as fp:
            raw_detection_meta_dict = json.load(fp)

        self.detection_meta_dict = dict()
        # Bad data list (invalid GT or data)
        if "2021" in config["dataset_folder"]:
            bad_list = ["D_M_D00000113", "D_M_D00000243", "D_M_D00000244", "D_M_D00000249"]
        else:
            bad_list = ["T_B_D00000501", "P_B_D00000379", ]

        for anno in raw_detection_meta_dict["annotations"]:
            folder_name = raw_detection_meta_dict["image_groups"][anno["image_group_id"] - 1]["group_name"]
            if folder_name in bad_list:
                continue

            if folder_name not in self.detection_meta_dict:
                self.detection_meta_dict[folder_name] = [anno]
            else:
                self.detection_meta_dict[folder_name].append(anno)

            if len(self.detection_meta_dict[folder_name]) >= 3 or len(self.detection_meta_dict[folder_name]) <= 0:
                print("Data Load Error: Folder {} has {} instances".format(folder_name,
                                                                           len(self.detection_meta_dict[folder_name])))

        raw_folders = sorted(glob.glob(os.path.join(config["dataset_folder"], dataset_type, "*D*")))
        self.folders = [f for f in raw_folders if os.path.basename(f) not in bad_list]
        # print("Making Train Dataset Object ... {} Instances".format(len(self.folders)))
        # print("raw_detection_meta keys: ", raw_detection_meta_dict.keys())
        # print("type: ", raw_detection_meta_dict['categories'])
        # exit()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        data_folder = self.folders[index]

        '''
        Read Laser Images
        '''
        laser_images_01 = sorted(glob.glob(os.path.join(data_folder, "reflection_image_*_C01.png")))
        laser_images_02 = sorted(glob.glob(os.path.join(data_folder, "reflection_image_*_C02.png")))

        W, H = self.config["laser_size"]  # target laser image size for input

        one_frame = cv2.imread(laser_images_01[0])  # to get original laser image size
        if one_frame is None:
            print("Laser PNG Error: {}".format(data_folder))
            return torch.zeros(size=(1,)), torch.zeros(size=(1,))

        l_H, l_W, _ = one_frame.shape
        h = int(round(l_H / 3))  # Naive pre-processing for cropping background
        try:
            laser_images_01 = [np.transpose(cv2.imread(path)[:-h], (1, 0, 2))[:, ::-1] for path in laser_images_01]
            laser_images_02 = [np.transpose(cv2.imread(path)[:-h], (1, 0, 2))[:, ::-1] for path in laser_images_02]
        except TypeError:
            print("Laser PNG Error: {}".format(data_folder))
            return None, None

        laser_images_01 = [cv2.resize(image, (W, H)) for image in laser_images_01]
        laser_images_02 = [cv2.resize(image, (W, H)) for image in laser_images_02]

        # Grid_H, Grid_W, Image_H, Image_W, 3
        laser_images_01 = np.transpose(np.reshape(laser_images_01, (5, 5, H, W, 3)), (1, 0, 2, 3, 4))
        laser_images_02 = np.transpose(np.reshape(laser_images_02, (5, 5, H, W, 3)), (1, 0, 2, 3, 4))

        # 2, G_H, G_W, I_H, I_W, 3
        laser_images = np.stack([laser_images_01, laser_images_02], axis=0)

        '''
        Load RF Data
        '''
        try:
            rf_data = list()
            rf_file_list = sorted(glob.glob(os.path.join(data_folder, "RF_*.npy")))  # RF_0, ..., RF_19
            for rf in rf_file_list:  # rf npy load
                temp_raw_rf = np.load(rf)
                temp_raw_rf = temp_raw_rf[:, :, 200:-312]

                # ----- normalization ------
                for i in range(temp_raw_rf.shape[0]):
                    for j in range(temp_raw_rf.shape[1]):
                        stdev = np.std(temp_raw_rf[i, j])
                        mean = np.mean(temp_raw_rf[i, j])
                        temp_raw_rf[i, j] = (temp_raw_rf[i, j] - mean) / stdev

                temp_raw_rf = torch.tensor(temp_raw_rf).float()
                # ---------- Convert to 2D -----------

                temp_raw_rf = rearrange(temp_raw_rf, 'tx rx len -> (tx rx) len')
                # print(temp_raw_rf.shape)
                temp_raw_rf = rearrange(temp_raw_rf, '(x y) len -> x y len', x=4)
                # print(temp_raw_rf.shape)
                temp_raw_rf = rearrange(temp_raw_rf, 'x y (len1 len2) -> x (len1 y) len2', len2=128)
                # print(temp_raw_rf.shape)

                rf_data.append(temp_raw_rf)
            rf_data = torch.stack(rf_data, dim=0).mean(dim=0).permute(1, 2, 0)
        except:
            print("RF Input Error: {}".format(data_folder))
            return torch.zeros(size=(1,)), torch.zeros(size=(1,))

        '''
        Load Sound Data
        '''
        try:
            sound_raw_data = np.load(os.path.join(data_folder, 'WAVE.npy'))
        except:
            print("Sound Input Error: {}".format(data_folder))
            return torch.zeros(size=(1,)), torch.zeros(size=(1,))

        sound_data = self._waveform_to_stft(sound_raw_data).permute(1, 2, 0)

        '''
        Read RGB & Depth Images and Dection Annotions for GT
        '''
        rgb_image = cv2.imread(os.path.join(data_folder, "gt_rgb_image.png"))
        if rgb_image is None:
            print("RGB PNG Error: {}".format(data_folder))
            return torch.zeros(size=(1,)), torch.zeros(size=(1,))
        depth_image = cv2.imread(os.path.join(data_folder, "gt_depth_gray_image.png"), cv2.IMREAD_GRAYSCALE)
        if depth_image is None:
            print("Depth PNG Error: {}".format(data_folder))
            return torch.zeros(size=(1,)), torch.zeros(size=(1,))

        laser_images = ((np.array(laser_images, dtype=np.float32) / 255.0) - 0.5) * 2.0
        rgb_image = np.array(rgb_image, dtype=np.float32) / 255.0
        depth_image = np.array(depth_image, dtype=np.float32) / 255.0

        # number of instances, GT data
        # Maximum number of instances is 2
        detection_gt = np.zeros(dtype=np.float32, shape=(2, 6))
        rgb_h, rgb_w, _ = rgb_image.shape
        try:
            gt_annos = self.detection_meta_dict[os.path.basename(data_folder)]
        except KeyError:
            gt_annos = self.detection_meta_dict[os.path.basename(data_folder).replace("_D0", "/D0")]
        for a_i, anno in enumerate(gt_annos):
            bbox = anno["bbox"]
            bbox = [bbox[0] / rgb_w, bbox[1] / rgb_h,
                    bbox[0] / rgb_w + bbox[2] / rgb_w,
                    bbox[1] / rgb_h + bbox[3] / rgb_h]
            class_id = float(anno["category_id"]) - 1.0  # change 1-base to 0-base
            valid_flag = 1.0  # this indicates this GT information is valid (for 1 instance case, second GT is invalid)
            # GT format: valid_flag, topleft_x, topleft_y, bottomright_x, bottomright_y, class_id
            detection_gt[a_i] = np.array([valid_flag] + bbox + [class_id], dtype=np.float32)

        laser_images = torch.from_numpy(laser_images)
        rgb_image = torch.from_numpy(rgb_image)
        depth_image = torch.from_numpy(depth_image)
        detection_gt = torch.from_numpy(detection_gt)

        features = (laser_images, rf_data, sound_data)
        targets = (rgb_image, depth_image, detection_gt)

        # jeon
        targets = {}
        targets["boxes"] = detection_gt[:, 1:5]
        targets["labels"] = detection_gt[:, -1].long()
        #
        # # taeho
        # targets["orig_size"] = torch.as_tensor([int(rgb_h), int(rgb_w)])
        # targets["size"] = torch.as_tensor([int(rgb_h), int(rgb_w)])
        #
        # targets["image_id"] = torch.as_tensor(index)

        rgb_image, targets = self.transforms(rgb_image, targets)
        # print(rgb_image.shape, detection_gt.shape)
        # return rgb_image.permute(2, 0, 1), targets

        return features, targets

    def _waveform_to_stft(self, audio_waveform, split=False, n_fft=512, win_length=64):
        audio_waveform = np.moveaxis(audio_waveform, [0, 1, 2], [0, 2, 1])

        audio_stft = []

        for x in range(audio_waveform.shape[0]):  # audio_waveform.shape[0]
            stft_channel = []
            for y in range(audio_waveform.shape[1]):  # audio_waveform.shape[1]
                temp = audio_waveform[x][y]  # [4500:14401]
                temp = librosa.stft(temp, n_fft=n_fft, win_length=win_length, center=False)
                if split:
                    stft_channel.append(np.abs(temp))
                else:
                    audio_stft.append(np.abs(temp))
            if (split == 'True'): audio_stft.append(stft_channel)

        return torch.FloatTensor(np.asarray(audio_stft))


# def make_nlos_transforms(image_set):
#     normalize = T.Compose([
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
#
#     if image_set == 'training':
#         return T.Compose([
#             T.RandomHorizontalFlip(),
#             T.RandomSelect(
#                 T.RandomResize(scales, max_size=1333),
#                 T.Compose([
#                     T.RandomResize([400, 500, 600]),
#                     T.RandomSizeCrop(384, 600),
#                     T.RandomResize(scales, max_size=1333),
#                 ])
#             ),
#             normalize,
#         ])
#
#     if image_set == 'validation':
#         return T.Compose([
#             T.RandomResize([800], max_size=1333),
#             normalize,
#         ])
#
#     raise ValueError(f'unknown {image_set}')

def make_nlos_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'training':
        return T.Compose([
            T.RandomResize([256], max_size=1333),
            normalize,
        ])

    if image_set == 'validation':
        return T.Compose([
            T.RandomResize([256]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    dataset_config = dict()
    # original laser input full size: 1936 x 1216
    dataset_config["laser_size"] = (64, 128)  # W, H, which is the target size for input
    dataset_config["dataset_folder"] = args.data_path
    dataset = NlosDataset(dataset_config, dataset_type=image_set, transforms=make_nlos_transforms(image_set))

    return dataset


if __name__ == "__main__":
    dataset_config = dict()
    # original laser input full size: 1936 x 1216
    dataset_config["laser_size"] = (64, 128)  # W, H, which is the target size for input
    dataset_config["dataset_folder"] = os.path.join("/datasets/NLOS/2022")

    # dataset_type = {"training", "validation"}
    dataset = NlosDataset(dataset_config, dataset_type="validation", transforms=make_nlos_transforms("validation"))

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                            num_workers=2, drop_last=False,
                            pin_memory=True, prefetch_factor=2)

    for features, targets in tqdm(dataloader):
        # pass
        # print("features shape: ", features.shape)
        laser_images, rf_data, sound_data = features
        # print("targets: ", targets.keys())
        rgb_image, depth_image, detection_gt = targets

        print("features")
        print("laser_image shape: ", laser_images.shape)
        print("rf_data shape: ", rf_data.shape)
        print("sound data shape: ", sound_data.shape)
        print("\n")

        print("targets")
        print("rgb_image shape: ", rgb_image.shape)
        print("depth_image shape: ", depth_image.shape)
        print("detection_gt shape: ", detection_gt.shape)

        exit()
