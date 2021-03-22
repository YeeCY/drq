import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)

        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug


class FrameStackReplayBuffer(ReplayBuffer):
    """Only store unique frames to save memory

    """
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device, frame_stack):
        super().__init__(obs_shape, action_shape, capacity, image_pad, device)
        self.frame_stack = frame_stack

        self.stack_frame_dists = np.empty((capacity, self.frame_stack), dtype=np.int32)

        # (chongyi zheng): We need to set the final not_done = 0.0 to make sure the correct stack when the first
        #   observation is sampled. Note that empty array is initialized to be all zeros.
        # self.not_dones[-1] = 0.0

    def add(self, obs, action, reward, next_obs, done, done_no_max, stack_frame_dists=np.empty(0)):
        """
        (chongyi zheng): other_frame_dists are relative indices of the other frame from the current one
        """
        assert len(stack_frame_dists) == self.frame_stack, "Relative indices of stacked frames must be provided!"

        np.copyto(self.obses[self.idx], obs[-1 * self.obs_shape[0]:])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs[-1 * self.obs_shape[0]:])
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        np.copyto(self.stack_frame_dists[self.idx], stack_frame_dists)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        # Reconstruct stacked observations:
        #   If the sampled observation is the first frame of an episode, it is stacked with itself.
        #   Otherwise, we stack it with the previous frames.
        #   The most recently not_done indicator must be 0.0 for the first 'frame_stack' frames of an episode
        #       first_frame: obs = stack([first_frame, first_frame, first_frame])
        #       second_frame: obs = stack([first_frame, first_frame, second_frame])
        #       third_frame: obs = stack([first_frame, second_frame, third_frame])
        #       forth_frame: obs = stack([second_frame, third_frame, forth_frame])
        #       ...
        obses = []
        next_obses = []
        stack_frame_dists = self.stack_frame_dists[idxs]
        for sf_idx in range(self.frame_stack):
            obses.append(self.obses[stack_frame_dists[:, sf_idx] + idxs])
            next_obses.append(self.next_obses[stack_frame_dists[:, sf_idx] + idxs])
        obses = np.concatenate(obses, axis=1)
        next_obses = np.concatenate(next_obses, axis=1)
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)

        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug
