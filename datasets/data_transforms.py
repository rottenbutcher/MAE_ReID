import numpy as np
import torch
import random
import torch.nn.functional as F
import math

class PointcloudRotate(object):
    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            pc[i, :, :] = torch.matmul(pc[i], R)
        return pc

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
            
        return pc

class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc

class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()
            
        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc

class RandomHorizontalFlip(object):


  def __init__(self, upright_axis = 'z', is_temporal=False):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])


  def __call__(self, coords):
    bsize = coords.size()[0]
    for i in range(bsize):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(coords[i, :, curr_ax])
                    coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
    return coords
  


class PointcloudViewpointMasking(object):
    def __init__(self, viewpoint_mask_ratio=0.5, random_mask_ratio=0.4):
        self.viewpoint_mask_ratio = viewpoint_mask_ratio
        self.random_mask_ratio = random_mask_ratio

    def generate_mask_single(self, points):
        """
        Generates a viewpoint-based boolean mask for a single point cloud.
        Input: points (N, 3) tensor
        Output: mask (N,) boolean tensor (True means masked)
        """
        N, _ = points.shape

        # --- Viewpoint Generation ---
        elevation_angle = torch.tensor(0.0, device=points.device, dtype=points.dtype)
        while True:
            azimuth_angle_deg = torch.FloatTensor(1).uniform_(0, 360)
            if not ((azimuth_angle_deg >= 70 and azimuth_angle_deg <= 110) or \
                    (azimuth_angle_deg >= 250 and azimuth_angle_deg <= 290)):
                break
        azimuth_angle = torch.deg2rad(azimuth_angle_deg).to(points.device)
        xz_proj = torch.cos(elevation_angle)
        x = xz_proj * torch.cos(azimuth_angle)
        y = torch.sin(elevation_angle)
        z = xz_proj * torch.sin(azimuth_angle)
        viewpoint = torch.tensor([x, y, z], device=points.device, dtype=points.dtype).squeeze().unsqueeze(0) # (1, 3)

        # --- Masking ---
        centered_points = points - points.mean(dim=0, keepdim=True)
        dot_product = torch.sum(centered_points * viewpoint, dim=-1) # (N,)

        num_viewpoint_masked = int(N * self.viewpoint_mask_ratio)
        sorted_indices = torch.argsort(dot_product)
        viewpoint_masked_indices = sorted_indices[:num_viewpoint_masked]
        viewpoint_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
        viewpoint_mask[viewpoint_masked_indices] = True

        visible_indices = sorted_indices[num_viewpoint_masked:]
        num_remaining = N - num_viewpoint_masked
        num_random_masked = int(num_remaining * self.random_mask_ratio)
        random_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
        if num_remaining > 0 and num_random_masked > 0:
             shuffled_visible_indices = visible_indices[torch.randperm(num_remaining, device=points.device)]
             random_masked_indices = shuffled_visible_indices[:num_random_masked]
             random_mask[random_masked_indices] = True

        final_mask_single = viewpoint_mask | random_mask
        return final_mask_single # Should return (N,) boolean tensor

    # --- Original __call__ method (kept for compatibility) ---
    def __call__(self, pc):
        B, N, _ = pc.shape
        final_mask_batch = torch.zeros(B, N, dtype=torch.bool, device=pc.device)
        for i in range(B):
            # Calls the method defined above
            final_mask_batch[i] = self.generate_mask_single(pc[i])
        masked_pc = pc.clone()
        masked_pc[final_mask_batch] = 0.0
        return masked_pc

# --- Helper function (should be OUTSIDE the class) ---
def generate_two_viewpoint_masks(points_tensor, viewpoint_mask_ratio=0.5, random_mask_ratio=0.4):
    masker = PointcloudViewpointMasking(viewpoint_mask_ratio, random_mask_ratio)
    # This line causes the error if generate_mask_single is not defined correctly in the class
    mask1 = masker.generate_mask_single(points_tensor)
    masker = PointcloudViewpointMasking(viewpoint_mask_ratio, random_mask_ratio)
    mask2 = masker.generate_mask_single(points_tensor)
    return mask1, mask2