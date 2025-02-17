import math
import random

import numpy as np
import torch
from scipy.linalg import expm, norm
from torchvision import transforms as transforms


class TrainSetTransform:
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            t = [RandomRotation(max_theta=5, axis=np.array([0, 0, 1])),
                 RandomFlip([0.25, 0.25, 0.])]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 2:
            # Augmentations with random rotation around z-axis
            t = [RandomRotation(max_theta=360, axis=np.array([0, 0, 1])),
                 RandomFlip([0.25, 0.25, 0.])]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 0:
            # No augmentation
            self.transform = None
        elif self.aug_mode == 7:
            # No augmentation
            t = [RandomRotation(max_theta=5, axis=np.array([0, 0, 1]))]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 8:
            # No augmentation
            t = [RandomFlip([0.25, 0.25, 0.])]
            self.transform = transforms.Compose(t)
        # elif self.aug_mode == 27 or self.aug_mode == 28 or self.aug_mode == 29 or self.aug_mode > 30:
        #     t = [RandomFlip([0.25, 0.25, 0.])]
        #     self.transform = transforms.Compose(t)
        else:
            self.transform = None
        

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class DA2best:
    def __init__(self, p_both=0.4, p_each = 0.5, scale=(0.02, 0.5), ratio=(0.3, 3.3), min=0.8, max=1.2):
        self.p_both = p_both
        self.p_each = p_each
        self.scale = scale
        self.ratio = ratio
        self.scale_xy = max - min
        self.bias = min

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h
    
    def remove_random_block(self, coords):
        x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
        mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
        coords[mask] = torch.zeros_like(coords[mask])
        return coords

    def random_scale_xy(self, coords):
        s = self.scale_xy * np.random.rand(1) + self.bias
        coords[..., :2] = coords[..., :2] * s
        return coords
    

    def __call__(self, coords):
        if random.random() < self.p_both:
            if random.random() <= self.p_each:
                coords = self.remove_random_block(coords)
            else:
                coords = self.random_scale_xy(coords)

        return coords




class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=None):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        # plot original point cloud wit open3d
        """
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        o3d.visualization.draw_geometries([pcd])
        """
        
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        random_value = np.random.rand(1)
        # random signo positivo o negativo
        #signo = np.random.choice([-1, 1])
        R = self._M(axis, (np.pi * self.max_theta / 180.) * 2. * (random_value - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180.) * 2. * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n

        # plot rotated point cloud with open3d
        """
        pcd_rotated = o3d.geometry.PointCloud()
        pcd_rotated.points = o3d.utility.Vector3dVector(coords)
        o3d.visualization.draw_geometries([pcd_rotated])
        """

        return coords
    
class RandomRotation_mod:
    def __init__(self, p=0.5, axis=None, max_theta=180, max_theta2=None):
        self.p = p
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        # plot original point cloud wit open3d
        """
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        o3d.visualization.draw_geometries([pcd])
        """
        if random.random() < self.p:
            if self.axis is not None:
                axis = self.axis
            else:
                axis = np.random.rand(3) - 0.5
            random_value = np.random.rand(1)
            # random signo positivo o negativo
            signo = np.random.choice([-1, 1])
            # random angle bewteen 0 and and self.max_theta
            #angle = np.random.uniform(0, self.max_theta) * signo
            #print("Angle in degrees: ", angle)
            # angle to radians
            #angle = np.radians(angle)
            angle_rad = (np.pi * signo * self.max_theta / 180.) * 2. * (random_value - 0.5)
            # angle radian to degrees
            #angle_deg = np.degrees(angle_rad)
            #print("Angle in degrees: ", angle_deg)
            R = self._M(axis, angle_rad)
            #print("R: ", R)
            if self.max_theta2 is None:
                coords = coords @ R
            else:
                R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180.) * 2. * (np.random.rand(1) - 0.5))
                coords = coords @ R @ R_n

            # plot rotated point cloud with open3d
            """
            pcd_rotated = o3d.geometry.PointCloud()
            pcd_rotated.points = o3d.utility.Vector3dVector(coords)
            o3d.visualization.draw_geometries([pcd_rotated])
            """

        return coords


class RandomTranslation:
    def __init__(self, max_delta=0.05, p=1.1):
        self.max_delta = max_delta
        self.p = p

    def __call__(self, coords):
        if random.random() < self.p:
            trans = self.max_delta * np.random.randn(1, 3)
            return coords + trans.astype(np.float32)
        else:
            return coords

class RandomTranslationB:
    def __init__(self, max_delta=0.05, p=1.1):
        self.max_delta = max_delta
        self.p = p

    def __call__(self, coords):
        if random.random() < self.p:
                
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(coords)
            # o3d.visualization.draw_geometries([pcd])
        
            trans = self.max_delta * np.random.randn(1, 3) * np.array([1, 1, 0])

            # pcd_mod = o3d.geometry.PointCloud()
            # pcd_mod.points = o3d.utility.Vector3dVector(coords + trans.astype(np.float32))
            # o3d.visualization.draw_geometries([pcd, pcd_mod])

            return coords + trans.astype(np.float32)
        else:
            return coords

class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1., probability=1.1):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p
        self.probability = probability

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        if random.random() < self.probability:
            sample_shape = (e.shape[0],)
            if self.p < 1.:
                # Create a mask for points to jitter
                m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
                mask = m.sample(sample_shape=sample_shape)
            else:
                mask = torch.ones(sample_shape, dtype=torch.int64 )

            mask = mask == 1
            jitter = self.sigma * torch.randn_like(e[mask])

            if self.clip is not None:
                jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

            e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r, p=1.1):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
            self.p = p
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        if random.random() < self.p:
            n = len(e)
            if self.r_min is None:
                r = self.r_max
            else:
                # Randomly select removal ratio
                r = random.uniform(self.r_min, self.r_max)

            mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
            e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        """
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        """
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        """
        pcd_tranformed = o3d.geometry.PointCloud()
        pcd_tranformed.points = o3d.utility.Vector3dVector(coords)
        pcd_tranformed.translate([1.5, 0.0, 0.0])
        o3d.visualization.draw_geometries([pcd, pcd_tranformed])
        """
        return coords
    
class MoveRandomBlock:
    """
    Randomly move part of the point cloud. 
    Moves a fronto-parallel cuboid to a random position towards the center of the point cloud.
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), max_move=0.2):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.max_move = max_move  # Maximum distance to move the block

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        """
        Apply a random movement to a block of points within the cloud.
        """
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)  # Fronto-parallel cuboid to move
            # Find the center of the point cloud
            cloud_center = torch.mean(coords, dim=0)

            # Define mask for points inside the block to be moved
            mask = (x < coords[..., 0]) & (coords[..., 0] < x + w) & (y < coords[..., 1]) & (coords[..., 1] < y + h)

            # Generate random movement vector towards the center of the point cloud
            move_vector = (cloud_center - torch.mean(coords[mask], dim=0)) * self.max_move * torch.rand(1).item()

            # Apply movement to the selected block
            signo = np.random.choice([-1, 1])
            coords[mask] = coords[mask] + move_vector * signo

        return coords
    
class MoveRandomBlock2:
    """
    Randomly move part of the point cloud. 
    Moves a fronto-parallel cuboid to a random position towards the center of the point cloud.
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), max_move=0.2):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.max_move = max_move  # Maximum distance to move the block
    
    def get_params_v2(self, coords):
        # Find point cloud 3D bounding box above a certain height
        # z min height of coords[:, 2]
        min_z = torch.min(coords[:, 2])
        # z max height of coords[:, 2]
        max_z = torch.max(coords[:, 2])
        # Find point cloud 3D bounding box above a certain height
        height = min_z + random.uniform(0.5, 1.0) * (max_z - min_z)
        # print('Min z: ', min_z)
        # print('Max z: ', max_z)
        # print("Height: ", height)
        mask = coords[:, 2] > height
        coords = coords[mask]
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h
        

    def __call__(self, coords):
        """
        Apply a random movement to a block of points within the cloud.
        """
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(coords)

        if random.random() < self.p:
            # x, y, w, h = self.get_params(coords)  # Fronto-parallel cuboid to move
            x, y, w, h = self.get_params_v2(coords)  # Fronto-parallel cuboid to move
            # Find the center of the point cloud
            cloud_center = torch.mean(coords, dim=0)

            # Define mask for points inside the block to be moved
            mask = (x < coords[..., 0]) & (coords[..., 0] < x + w) & (y < coords[..., 1]) & (coords[..., 1] < y + h)

            # Generate random movement vector towards the center of the point cloud
            move_vector = (cloud_center - torch.mean(coords[mask], dim=0)) * self.max_move * torch.rand(1).item()

            # Apply movement to the selected block
            signo = np.random.choice([-1, 1])
            coords[mask] = coords[mask] + move_vector * signo

            # pcd_tranformed = o3d.geometry.PointCloud()
            # pcd_tranformed.points = o3d.utility.Vector3dVector(coords)
            # pcd_tranformed.translate([2.5, 0.0, 0.0])
            # o3d.visualization.draw_geometries([pcd, pcd_tranformed])

        return coords

class RandomScale:
    def __init__(self, min, max, p=1.1):
        self.scale = max - min
        self.bias = min
        self.p = p

    def __call__(self, coords):
        if random.random() < self.p:
            s = self.scale * np.random.rand(1) + self.bias
            return coords * s
        else:
            return coords

class RandomScaleXY:
    def __init__(self, min, max, p=1.1):
        self.scale = max - min
        self.bias = min
        self.p = p
    
    def __call__(self, coords):
        if random.random() < self.p:
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(coords)
            s = self.scale * np.random.rand(1) + self.bias
            coords[..., :2] = coords[..., :2] * s

            # pcd_tranformed = o3d.geometry.PointCloud()
            # pcd_tranformed.points = o3d.utility.Vector3dVector(coords)
            # #pcd_tranformed.translate([1.5, 0.0, 0.0])
            # o3d.visualization.draw_geometries([pcd, pcd_tranformed])

        return coords

class RandomShear:
    def __call__(self, coords):
        T = np.eye(3) + np.random.randn(3, 3)
        return coords @ T
    

class ElasticDistortion:
    def __init__(self, granularity=0.1, magnitude=0.1):
        self.granularity = granularity
        self.magnitude = magnitude

    def __call__(self, coords):
        blur = self.granularity
        noise = np.random.randn(*coords.shape) * self.magnitude
        smoothed_noise = np.convolve(noise, np.ones((3,)) / 3, mode='same')
        coords += smoothed_noise * blur
        return coords

class RandomOcclusion:
    def __init__(self, p=0.5, block_size=0.2):
        self.p = p
        self.block_size = block_size

    def __call__(self, coords):
        if random.random() < self.p:
            # coords from torch to numpy
            coords = coords.numpy()
            # min coords of the bounding box
            min_coords = np.min(coords, axis=0)
            # max coords of the bounding box
            max_coords = np.max(coords, axis=0)
            occlusion_center = min_coords + (max_coords - min_coords) * np.random.rand(3)
            mask = np.linalg.norm(coords - occlusion_center, axis=1) < self.block_size
            # coords from numpy to torch
            coords = torch.from_numpy(coords)
            coords[mask] = torch.zeros_like(coords[mask])

            
        return coords

class AddRandomNoise:
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, coords):
        noise = self.sigma * np.random.randn(*coords.shape)
        noise = np.clip(noise, -self.clip, self.clip)
        return coords + noise
    
class RandomDropout:
    def __init__(self, drop_rate=0.1):
        self.drop_rate = drop_rate

    def __call__(self, coords):
        mask = np.random.rand(len(coords)) > self.drop_rate
        return coords[mask]