import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T

def get_train_transform():
    transforms = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    return T.Compose(transforms)

def get_val_transform():
    transforms = [
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    return T.Compose(transforms)

def get_cifar100(root, args, val_ratio=0.1, val_from_train_ratio=0.5, balanced_val=False, val_samples_per_class=None):
    """
    Args:
        root: 数据集根目录
        args: 参数对象
        val_ratio: 验证集总比例（默认0.1=10%），当balanced_val=True时，用于计算总验证集大小
        val_from_train_ratio: 验证集中从训练集采样的比例（默认0.5=50%，即5%从训练集，5%从剩余数据）
        balanced_val: 是否创建平衡验证集（每个类别相同数量的样本）
        val_samples_per_class: 平衡验证集中每个类别的样本数（如果为None，则根据val_ratio计算）
    """
    transform_train = get_train_transform()
    transform_val = get_val_transform()

    # 方案C: 混合采样（或平衡验证集）
    # 1. 创建完整的长尾训练集（用于后续划分）
    full_train_dataset = CIFAR100_train_full(root, args, imb_ratio=args.imb_ratio, train=True, transform=transform_train)
    
    # 2. 从训练集和剩余数据中混合采样验证集（或创建平衡验证集）
    train_dataset, val_dataset = split_train_val_mixed(
        full_train_dataset, val_ratio=val_ratio, val_from_train_ratio=val_from_train_ratio, 
        args=args, balanced_val=balanced_val, val_samples_per_class=val_samples_per_class
    )
    
    # 3. 测试集（不变）
    test_dataset = CIFAR100_val(root, transform=transform_val)
    
    # 打印验证集分布信息
    if balanced_val:
        val_targets_np = np.array(val_dataset.targets)
        val_samples_per_cls = [np.sum(val_targets_np == i) for i in range(100)]
        min_val_samples = min(val_samples_per_cls)
        max_val_samples = max(val_samples_per_cls)
        avg_val_samples = np.mean(val_samples_per_cls)
        print(f"#Train: {len(train_dataset)}, #Val: {len(val_dataset)} (balanced, avg {avg_val_samples:.1f} samples per class, range [{min_val_samples}, {max_val_samples}]), #Test: {len(test_dataset)}")
        if min_val_samples < val_samples_per_class:
            print(f"Warning: Some classes have fewer than target {val_samples_per_class} validation samples. "
                  f"Minimum: {min_val_samples}, Maximum: {max_val_samples}")
    else:
        print(f"#Train: {len(train_dataset)}, #Val: {len(val_dataset)}, #Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

class CIFAR100_train_full(torchvision.datasets.CIFAR100):
    """完整的长尾训练集（用于后续划分）"""
    def __init__(self, root, args, imb_type='exp', imb_ratio=100, train=True, transform=None, target_transform=None, download=True):

        super(CIFAR100_train_full, self).__init__(root, train=train, transform=transform,
                                             target_transform=target_transform,
                                             download=download)
        self.args = args
        self.cls_num = 100
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, 1. / imb_ratio)
        self.transform_train = transform
        self.num_per_cls_dict = dict()
        # 保存原始数据索引，用于后续划分
        self.original_data = self.data.copy()
        self.original_targets = np.array(self.targets, dtype=np.int64)
        self.gen_imbalanced_data(self.img_num_list)
        # 保存已使用的索引
        self.used_indices = set()
        for the_class in range(self.cls_num):
            idx = np.where(self.original_targets == the_class)[0]
            used_idx = idx[:self.num_per_cls_dict[the_class]]
            self.used_indices.update(used_idx.tolist())

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)

        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # print(selec_idx)
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_category(self, target):
        per_class_num = self.num_per_cls_dict[target]
        if per_class_num > 100: return 'Many'
        if 100 >= per_class_num >= 20: return 'Medium'
        if per_class_num < 20: return 'Few'

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img = self.transform_train(img)

        return img, target, index


class CIFAR100_train(torch.utils.data.Dataset):
    """从完整训练集划分出的训练集"""
    def __init__(self, data, targets, transform=None, num_per_cls_dict=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.num_per_cls_dict = num_per_cls_dict or {}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index
    
    @property
    def img_num_list(self):
        """返回每个类别的样本数列表"""
        return [self.num_per_cls_dict.get(i, 0) for i in range(100)]
    
    def get_category(self, target):
        per_class_num = self.num_per_cls_dict.get(target, 0)
        if per_class_num > 100: return 'Many'
        if 100 >= per_class_num >= 20: return 'Medium'
        if per_class_num < 20: return 'Few'


class CIFAR100_val_split(torch.utils.data.Dataset):
    """从完整训练集划分出的验证集"""
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index


def split_train_val_mixed(full_dataset, val_ratio=0.1, val_from_train_ratio=0.5, args=None, seed=42, balanced_val=False, val_samples_per_class=None):
    """
    方案C: 混合采样验证集
    从当前训练集采样 val_ratio * val_from_train_ratio，从剩余数据采样 val_ratio * (1 - val_from_train_ratio)
    
    如果 balanced_val=True，创建平衡验证集（每个类别相同数量的样本）
    
    Args:
        full_dataset: 完整的长尾训练集 (CIFAR100_train_full)
        val_ratio: 验证集总比例（默认0.1=10%），当balanced_val=True时，用于计算总验证集大小
        val_from_train_ratio: 验证集中从训练集采样的比例（默认0.5=50%）
        args: 参数对象
        seed: 随机种子
        balanced_val: 是否创建平衡验证集（每个类别相同数量的样本）
        val_samples_per_class: 平衡验证集中每个类别的样本数（如果为None，则根据val_ratio计算）
    
    Returns:
        train_dataset: 训练集
        val_dataset: 验证集
    """
    np.random.seed(seed)
    
    # 获取每个类别的样本数
    img_num_list = full_dataset.img_num_list
    num_per_cls_dict = full_dataset.num_per_cls_dict
    original_targets = full_dataset.original_targets
    original_data = full_dataset.original_data
    used_indices = full_dataset.used_indices
    
    # 按类别划分
    train_data = []
    train_targets = []
    val_data = []
    val_targets = []
    
    if balanced_val:
        # 平衡验证集：每个类别相同数量的样本（默认与测试集一致，每个类别100个样本）
        if val_samples_per_class is None:
            # 默认使用100个样本（与测试集一致）
            val_samples_per_class = 100
        
        # 确保不超过CIFAR-100每个类别的最大可用样本数（原始训练集每个类别500个样本）
        # 考虑到长尾分布，某些类别可能样本数较少，但验证集应该尽可能接近目标值
        print(f"Creating balanced validation set: target {val_samples_per_class} samples per class (matching test set distribution)")
        
        for cls_idx in range(full_dataset.cls_num):
            # 获取该类别的所有样本索引
            cls_mask = (original_targets == cls_idx)
            cls_indices = np.where(cls_mask)[0]
            
            # 分离已使用的索引和未使用的索引
            used_cls_indices = [idx for idx in cls_indices if idx in used_indices]
            unused_cls_indices = [idx for idx in cls_indices if idx not in used_indices]
            
            # 计算该类别的验证集样本数（平衡：每个类别尽可能接近目标数量）
            # 目标：每个类别val_samples_per_class个样本（默认100，与测试集一致）
            cls_val_target = val_samples_per_class
            
            # 计算该类别的总可用样本数（训练集 + 剩余数据）
            total_available = len(used_cls_indices) + len(unused_cls_indices)
            
            # 如果总可用样本数不足目标值，则使用所有可用样本
            if total_available < cls_val_target:
                cls_val_total = total_available
                print(f"Warning: Class {cls_idx} has only {total_available} available samples, less than target {cls_val_target}. Using all available samples.")
            else:
                cls_val_total = cls_val_target
            
            # 优先从剩余数据采样（因为剩余数据是原始平衡数据，更适合验证集）
            # 如果剩余数据不够，再从训练集补充
            cls_val_from_unused = min(cls_val_total, len(unused_cls_indices))
            cls_val_from_train = max(0, cls_val_total - cls_val_from_unused)
            
            # 确保训练集每个类别至少保留1个样本（如果可能）
            if cls_val_from_train >= len(used_cls_indices):
                # 如果需要的训练集样本数超过可用数，调整采样策略
                cls_val_from_train = max(0, len(used_cls_indices) - 1)  # 至少保留1个训练样本
                cls_val_from_unused = min(cls_val_total - cls_val_from_train, len(unused_cls_indices))
                cls_val_total = cls_val_from_train + cls_val_from_unused
            
            cls_train_num = len(used_cls_indices) - cls_val_from_train  # 新训练集样本数
            
            # 随机打乱并划分
            np.random.shuffle(used_cls_indices)
            train_indices = used_cls_indices[:cls_train_num]
            val_from_train_indices = used_cls_indices[cls_train_num:cls_train_num + cls_val_from_train]
            
            # 从剩余数据采样
            val_from_unused_indices = []
            if cls_val_from_unused > 0 and len(unused_cls_indices) > 0:
                np.random.shuffle(unused_cls_indices)
                val_from_unused_indices = unused_cls_indices[:cls_val_from_unused]
            
            # 添加到训练集
            train_data.append(original_data[train_indices])
            train_targets.extend([cls_idx] * cls_train_num)
            
            # 添加到验证集（从训练集采样）
            if len(val_from_train_indices) > 0:
                val_data.append(original_data[val_from_train_indices])
                val_targets.extend([cls_idx] * len(val_from_train_indices))
            
            # 添加到验证集（从剩余数据采样）
            if len(val_from_unused_indices) > 0:
                val_data.append(original_data[val_from_unused_indices])
                val_targets.extend([cls_idx] * len(val_from_unused_indices))
    else:
        # 原始的长尾验证集采样策略
        for cls_idx in range(full_dataset.cls_num):
            # 获取该类别的所有样本索引
            cls_mask = (original_targets == cls_idx)
            cls_indices = np.where(cls_mask)[0]
            
            # 分离已使用的索引和未使用的索引
            used_cls_indices = [idx for idx in cls_indices if idx in used_indices]
            unused_cls_indices = [idx for idx in cls_indices if idx not in used_indices]
            
            # 计算该类别的验证集样本数
            cls_total = len(used_cls_indices)  # 当前训练集中的样本数
            cls_val_from_train = max(1, int(cls_total * val_ratio * val_from_train_ratio))  # 从训练集采样
            cls_train_num = cls_total - cls_val_from_train  # 新训练集样本数
            
            # 从剩余数据采样（如果有）
            cls_val_from_unused = 0
            if len(unused_cls_indices) > 0:
                # 计算需要从剩余数据采样的数量
                target_val_from_unused = max(1, int(cls_total * val_ratio * (1 - val_from_train_ratio)))
                cls_val_from_unused = min(target_val_from_unused, len(unused_cls_indices))
            
            # 随机打乱并划分
            np.random.shuffle(used_cls_indices)
            train_indices = used_cls_indices[:cls_train_num]
            val_from_train_indices = used_cls_indices[cls_train_num:cls_train_num + cls_val_from_train]
            
            # 从剩余数据采样
            val_from_unused_indices = []
            if cls_val_from_unused > 0 and len(unused_cls_indices) > 0:
                np.random.shuffle(unused_cls_indices)
                val_from_unused_indices = unused_cls_indices[:cls_val_from_unused]
            
            # 添加到训练集
            train_data.append(original_data[train_indices])
            train_targets.extend([cls_idx] * cls_train_num)
            
            # 添加到验证集（从训练集采样）
            if len(val_from_train_indices) > 0:
                val_data.append(original_data[val_from_train_indices])
                val_targets.extend([cls_idx] * len(val_from_train_indices))
            
            # 添加到验证集（从剩余数据采样）
            if len(val_from_unused_indices) > 0:
                val_data.append(original_data[val_from_unused_indices])
                val_targets.extend([cls_idx] * len(val_from_unused_indices))
    
    # 合并数据
    if train_data:
        train_data = np.vstack(train_data)
    else:
        train_data = np.array([])
    
    if val_data:
        val_data = np.vstack(val_data)
    else:
        val_data = np.array([])
    
    # 创建数据集对象
    train_dataset = CIFAR100_train(
        train_data, train_targets, 
        transform=full_dataset.transform_train,
        num_per_cls_dict={k: len([t for t in train_targets if t == k]) 
                         for k in range(full_dataset.cls_num)}
    )
    
    val_dataset = CIFAR100_val_split(
        val_data, val_targets,
        transform=get_val_transform()
    )
    
    return train_dataset, val_dataset


class CIFAR100_val(torchvision.datasets.CIFAR100):
    def __init__(self, root, transform=None, indexs=None,
                 target_transform=None, download=True):
        super(CIFAR100_val, self).__init__(root, train=False, transform=transform, target_transform=target_transform,
                                           download=download)

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index