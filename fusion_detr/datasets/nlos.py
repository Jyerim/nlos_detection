import os
import numpy as np
import torch
import cv2
import glob
import json
from torchvision import transforms as TT
import librosa
from einops import rearrange, reduce, repeat
from functools import partial
from torch.utils.data import Dataset
import datasets.transforms as T
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import ctypes

class NlosDataset(Dataset):

    def __init__(self, config, dataset_type, transforms=None, sensor=None):
        
        self.sensor = sensor
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
            bad_list = ["T_B_D00000501", "P_B_D00000379", "P_D00000198","P_D00000319"]

        for anno in raw_detection_meta_dict["annotations"]:
            folder_name = raw_detection_meta_dict["image_groups"][anno["image_group_id"] - 1]["group_name"]
            bad_group_name = [f_name.replace("_D0","/D0") for f_name in bad_list]
            if folder_name in bad_group_name:
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
        if dataset_type == "training":
            print("Making Train Dataset Object ... {} Instances".format(len(self.folders)))
            '''caching laser data on memory'''
            # shared_array_base = mp.Array(ctypes.c_float,3159*2*3*5*5*128*64)
            # shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            # shared_array = shared_array.reshape(3159,6,5,5,128,64)
            # self.shared_array = torch.from_numpy(shared_array)
            # self.use_cache = False
            '''caching laser data on memory'''
        else:
            print("Making Val Dataset Object ... {} Instances".format(len(self.folders)))
            '''caching laser data on memory'''
            # shared_array_base = mp.Array(ctypes.c_float,1345*2*3*5*5*128*64)
            # shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            # shared_array = shared_array.reshape(1345,6,5,5,128,64)
            # self.shared_array = torch.from_numpy(shared_array)
            # self.use_cache = False
            '''caching laser data on memory'''
        
        
    def __len__(self):
        return len(self.folders)
    
    '''caching laser data on memory'''
    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    '''caching laser data on memory'''
    
    def rf_working(self,rf):

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

        return temp_raw_rf
    
    def laser_working(self, path, l):
        
        l_H, l_W = l
        temp = l_W - l_H
        crop_start = int(round(temp / 2))
        h = int(round(l_H / 2))

        return np.transpose(cv2.imread(path)[:-h, crop_start:crop_start+h], (1, 0, 2))[:, ::-1]
    
    def __getitem__(self, index):
        data_folder = self.folders[index]
        '''
        Read Laser Images
        '''
        # if not self.use_cache: # caching laser data on memory
        laser_images = np.load(os.path.join(data_folder,'reflection_images.npy'))
        laser_images = torch.from_numpy(np.array(laser_images, dtype=np.float32))
        V, G_H, G_W, I_H, I_W, C = laser_images.shape
        laser_images = torch.permute(laser_images,(0, 5, 1, 2, 3, 4)).reshape(-1, I_H, I_W)
        mean, std = laser_images.mean([1,2]), laser_images.std([1,2])
        laser_norm = TT.Normalize(mean, std)
        laser_images = laser_norm(laser_images)
        if self.sensor == 'laser_4d':
            laser_images = laser_images.reshape(-1,G_H,G_W,I_H,I_W)
        # self.shared_array[index] = laser_images # caching laser data on memory
        
        # laser_images = self.shared_array[index] # caching laser data on memory

        
        '''
        Load RF Data
        '''
        
        try:
            rf_file_list = sorted(glob.glob(os.path.join(data_folder, "RF_*.npy")))  # RF_0, ..., RF_19
            pool = ThreadPool(len(rf_file_list))
            rf_data = pool.map(self.rf_working, rf_file_list)
            pool.close()
            pool.join()
            rf_data = torch.stack(rf_data, dim=0).mean(dim=0)
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

        sound_data = self._waveform_to_stft(sound_raw_data)
        mean, std = sound_data.mean([1,2]), sound_data.std([1,2])
        sound_norm = TT.Normalize(mean, std)
        sound_data = sound_norm(sound_data)


        '''
        Read RGB & Depth Images and Dection Annotions for GT : 
        '''
        rgb_image = Image.open(os.path.join(data_folder, "gt_rgb_image.png")).convert('RGB')
        if rgb_image is None:
            print("RGB PNG Error: {}".format(data_folder))
            return torch.zeros(size=(1,)), torch.zeros(size=(1,))

        w, h = rgb_image.size
        try:
            gt_annos = self.detection_meta_dict[os.path.basename(data_folder)]
        except KeyError:
            gt_annos = self.detection_meta_dict[os.path.basename(data_folder).replace("_D0", "/D0")]

        boxes = [ann["bbox"] for ann in gt_annos]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [int(ann["category_id"]) for ann in gt_annos]
        classes = torch.tensor(classes,dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        
        targets = {}
        targets["boxes"] = boxes
        targets["labels"] = classes
        targets["orig_size"] = torch.as_tensor([int(h),int(w)])
        targets["size"] = torch.as_tensor([int(h),int(w)])
        targets["image_group_id"] = torch.as_tensor(gt_annos[0]['image_group_id'])
        if self.transforms is not None:
            rgb_image, targets = self.transforms(rgb_image,targets)
        
        if self.sensor == 'laser' or self.sensor == 'laser_4d':
            return rgb_image, laser_images, targets
        elif self.sensor == 'rf':
            return rgb_image, rf_data, targets
        elif self.sensor == 'sound':
            return rgb_image, sound_data, targets
    

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

def make_nlos_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == 'training':
        return T.Compose([
            normalize,
        ])

    if image_set == 'validation':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def plot_check(img,targets):
    group_id, bboxes, labels = targets['image_group_id'], targets['boxes'], targets['labels']
    img = img.numpy()
    img = 255.0 * (img - np.min(img) + 1e-10) / (1e-10 + np.max(img) - np.min(img))
    img = img.squeeze()
    img = img.transpose(1,2,0)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    for (x_c,y_c,w,h) in bboxes[0]:
        img = cv2.rectangle(img, (int(x_c - w/2),int(y_c - h/2)),(int(x_c+w/2),int(y_c+h/2)),(255,0,0),3)
    cv2.putText(img,str(labels),(0,20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1)
    cv2.imwrite(os.path.join('./check','{}.jpg'.format(str(group_id))),img)

def build(image_set, args):
    dataset_config = dict()
    # original laser input full size: 1936 x 1216
    dataset_config["laser_size"] = (64, 128)  # W, H, which is the target size for input
    dataset_config["dataset_folder"] = args.data_path
    dataset = NlosDataset(dataset_config, dataset_type=image_set, transforms = make_nlos_transforms(image_set), sensor=args.sensor)
    
    return dataset

if __name__ == "__main__":
    dataset_config = dict()
    # original laser input full size: 1936 x 1216
    dataset_config["laser_size"] = (64, 128)  # W, H, which is the target size for input
    dataset_config["dataset_folder"] = os.path.join("/project/2022")

    # dataset_type = {"training", "validation"}
    # dataset = NlosDataset(dataset_config, dataset_type='training')
    dataset = NlosDataset(dataset_config, dataset_type="validation")

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=1, drop_last=False,
                            pin_memory=True, prefetch_factor=2)

    for data_folder, features, targets in tqdm(dataloader):
        # plot_check(features,targets)
        print(data_folder,features.shape)
        # laser_images, rf_data, sound_data = features
        # rgb_image, depth_image, detection_gt = targets

        # print(laser_images.shape)
        # print(rf_data.shape)
        # print(sound_data.shape)
        # #
        # print(rgb_image.shape)
        # print(depth_image.shape)
        # print(detection_gt.shape)