import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2
import img_to_gamma
import pandas as pd


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root, source):
    # root:'./datasets/hmdb51_frames'
    # source:'./datasets/settings/hmdb51/train_rgb_split1.txt'
    if not os.path.exists(source):
        print("Setting file %s for hmdb51 dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                clip_path = os.path.join(root, line_info[0])  # 视频名称
                duration = int(line_info[1])  # 视频帧长
                target = int(line_info[2])  # 视频类别
                item = (clip_path, duration, target)
                clips.append(item)
    return clips  # (视频名称,帧长,标签)


def ReadSegmentRGB(
    path, offsets, new_height, new_width, new_length, is_color, name_pattern, duration
):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR  # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE  # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length + 1):
            loaded_frame_index = length_id + offset
            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = duration + 1
            frame_name = name_pattern % (moded_loaded_frame_index)
            frame_path = path + "/" + frame_name
            cv_img_origin = cv2.imread(frame_path, cv_read_flag)

            if cv_img_origin is None:
                print("Could not load file %s" % (frame_path))
                sys.exit()
                # TODO: error handling here
            if new_width > 0 and new_height > 0:
                # use OpenCV3, use OpenCV2.4.13 may have error
                cv2.imwrite("old_img.jpg", cv_img_origin)
                cv_img = cv2.resize(
                    cv_img_origin, (new_width, new_height), interpolation
                )
                cv2.imwrite("resized_image.jpg", cv_img)

            else:
                cv_img = cv_img_origin
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            sampled_list.append(cv_img)

    # clip_input = np.concatenate(sampled_list, axis=2)
    return sampled_list
    # cv2.imwrite('dark_image.jpg', clip_input)

    return clip_input


def ReadSegmentRGB_light(
    path,
    offsets,
    new_height,
    new_width,
    new_length,
    is_color,
    name_pattern,
    duration,
    gamma,
):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR  # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE  # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length + 1):
            loaded_frame_index = length_id + offset
            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = duration + 1
            frame_name = name_pattern % (moded_loaded_frame_index)
            frame_path = path + "/" + frame_name
            cv_img_origin = cv2.imread(frame_path, cv_read_flag)
            #####
            cv_img_origin = img_to_gamma.gamma_intensity_correction(
                cv_img_origin, gamma
            )
            #####
            if cv_img_origin is None:
                print("Could not load file %s" % (frame_path))
                sys.exit()
                # TODO: error handling here
            if new_width > 0 and new_height > 0:
                # use OpenCV3, use OpenCV2.4.13 may have error
                cv_img = cv2.resize(
                    cv_img_origin, (new_width, new_height), interpolation
                )
            else:
                cv_img = cv_img_origin

            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            sampled_list.append(cv_img)
    return sampled_list

    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


def make_dataset_from_csv(root, source):
    # root:'./datasets/hmdb51_frames'
    # source:'./datasets/settings/hmdb51/train_rgb_split1.txt'
    if not os.path.exists(source):
        print("Setting file %s for hmdb51 dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        temp = pd.read_csv(source)
        for i in range(temp.shape[0]):
            clip_path = os.path.join(root, temp.iloc[i]["Video"])
            duration = int(temp.iloc[i]["Duration"])
            target = int(temp.iloc[i]["ClassID"])
            item = (clip_path, duration, target)
            clips.append(item)

    return clips


class ARID(data.Dataset):

    def __init__(
        self,
        phase,
        root="../data/ARID/ARID_frames",
        source="../data/ARID/test.csv",
        modality="rgb",
        name_pattern=None,
        is_color=True,
        num_segments=8,
        new_length=2,
        new_width=224,
        new_height=224,
        transform=None,
        target_transform=None,
        video_transform=None,
        ensemble_training=False,
        gamma=1.8,
        # gamma=None,
    ):

        classes, class_to_idx = find_classes(root)
        clips = make_dataset_from_csv(root, source)
        self.gamma = gamma
        # clips:(视频名称, 帧长, 标签)

        if len(clips) == 0:
            raise (
                RuntimeError(
                    "Found 0 video clips in subfolders of: " + root + "\n"
                    "Check your data directory."
                )
            )

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips
        self.ensemble_training = ensemble_training

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "img_%05d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%05d"

        self.is_color = is_color
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

        # added by Hemant
        self.class_num = 11
        self.joint_num = 18
        self.joint_names = (
            "Nose",
            "Thorax",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
            "L_Shoulder",
            "L_Elbow",
            "L_Wrist",
            "R_Hip",
            "R_Knee",
            "R_Ankle",
            "L_Hip",
            "L_Knee",
            "L_Ankle",
            "R_Eye",
            "L_Eye",
            "R_Ear",
            "L_Ear",
        )  # openpose joint set
        # self.skeleton = ( (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,8), (8,9), (9,10), (1,11), (11,12), (12,13), (0,14), (0,15), (14,16), (15,17) )
        self.skeleton = (
            (1, 0),
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 5),
            (5, 6),
            (6, 7),
            (1, 8),
            (8, 9),
            (9, 10),
            (1, 11),
            (11, 12),
            (12, 13),
        )

    def __getitem__(self, index):
        path, duration, target = self.clips[index]
        # print("index : ", index)
        # print(f"path: {path}, duration: {duration}, target: {target}")
        duration = duration - 1
        average_duration = int(duration / self.num_segments)
        average_part_length = int(
            np.floor((duration - self.new_length) / self.num_segments)
        )

        offsets = []
        for seg_id in range(self.num_segments):
            if self.phase == "train":
                if average_duration >= self.new_length:
                    offset = random.randint(0, average_duration - self.new_length)
                    # offset=2,
                    # No +1 because randint(a,b) return a random integer N such that a <= N <= b.
                    offsets.append(offset + seg_id * average_duration)
                elif duration >= self.new_length:
                    offset = random.randint(0, average_part_length)
                    offsets.append(seg_id * average_part_length + offset)
                else:
                    increase = random.randint(0, duration)
                    offsets.append(0 + seg_id * increase)
            elif self.phase == "val":
                if average_duration >= self.new_length:
                    offsets.append(
                        int(
                            (average_duration - self.new_length + 1) / 2
                            + seg_id * average_duration
                        )
                    )
                elif duration >= self.new_length:
                    offsets.append(
                        int(
                            (
                                seg_id * average_part_length
                                + (seg_id + 1) * average_part_length
                            )
                            / 2
                        )
                    )
                else:
                    increase = int(duration / self.num_segments)
                    offsets.append(0 + seg_id * increase)
            else:
                print("Only phase train and val are supported.")

        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(
                path,
                offsets,
                self.new_height,
                self.new_width,
                self.new_length,
                self.is_color,
                self.name_pattern,
                duration,
            )
            clip_input_light = ReadSegmentRGB_light(
                path,
                offsets,
                self.new_height,
                self.new_width,
                self.new_length,
                self.is_color,
                self.name_pattern,
                duration,
                gamma=self.gamma,
            )
        else:
            print("No such modality %s" % (self.modality))

        if self.transform is not None:
            clip_input = self.transform(clip_input)
            clip_input_light = self.transform(clip_input_light)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input, clip_input_light = self.video_transform(
                clip_input, clip_input_light
            )

        clip_input = np.array(clip_input)
        clip_input_light = np.array(clip_input_light)
        clip_input = clip_input.transpose(0, 3, 1, 2).astype(
            np.float32
        )  # frame_num, channel_dim, height, width
        clip_input_light = clip_input_light.transpose(0, 3, 1, 2).astype(
            np.float32
        )  # frame_num, channel_dim, height, width
        # print("clip_input shape: ", clip_input.shape)
        # print("clip_input_light shape: ", clip_input_light.shape)
        inputs = {"dark_video": clip_input, "light_video": clip_input_light}
        targets = {"action_label": target}
        # meta_info = {'img_id': data['img_id']}
        meta_info = {"img_id": 1}  # dummy value
        return inputs, targets, meta_info

        return clip_input, clip_input_light, target

    def __len__(self):
        return len(self.clips)
