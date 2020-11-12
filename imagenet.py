import math
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm

from mmd_critic import Dataset, select_prototypes, select_criticisms


cwd = Path('.')
output_dir = cwd / 'output'
imagenet_root = Path('~/ILSVRC2012/')
split='train'
device = torch.device('cpu')

class_name = 'Blenheim spaniel'

gamma = None

num_prototypes = 32
num_criticisms = 10

kernel_type = 'local'
# kernel_type = 'global'

# regularizer = None
regularizer = 'logdet'
# regularizer = 'iterative'
use_image_embeddings = False
batch_size = 64

make_plots = True

print('==============')
print(f'imagenet_root:{imagenet_root.absolute()}')
print(f'output_dir:{output_dir.absolute()}')
print(f'target_class:{class_name}')
print(f'num_prototypes:{num_prototypes}')
print(f'num_criticisms:{num_criticisms}')
print(f'gamma:{gamma}')
print(f'kernel_type:{kernel_type}')
print(f'regularizer:{regularizer}')
print(f'make_plots:{make_plots}')
print('==============\n')

# torch.set_num_threads(64)

imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


if use_image_embeddings:
    # ====== Run using image embeddings, as in Section 5.2 of the paper
    print('Preparing data...', end='', flush=True)
    imagenet_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_stats["mean"], std=imagenet_stats["std"]),
        ]
    )


    class NormalizeInverse(transforms.Normalize):
        def __init__(self, mean, std) -> None:
            mean = torch.as_tensor(mean)
            std = torch.as_tensor(std)
            std_inv = 1 / (std + 1e-7)
            mean_inv = -mean * std_inv
            super().__init__(mean=mean_inv, std=std_inv)

        def __call__(self, t: torch.Tensor) -> torch.Tensor:
            return super().__call__(t.clone()).clamp(min=0.0, max=1.0)


    imagenet_inverse_normalize = NormalizeInverse(mean=imagenet_stats['mean'], std=imagenet_stats['std'])

    ds = datasets.ImageNet(root=imagenet_root, split='train', transform=imagenet_tfms)
    class_idx = ds.class_to_idx[class_name]
    ds = torch.utils.data.Subset(ds, torch.where(torch.tensor(ds.targets) == class_idx)[0])
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    model = models.resnet50(pretrained=True)
    
    class Identity(torch.nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
            
        def forward(self, x):
            return x
            
    model.fc = Identity()

    embeddings = []
    for x, y in tqdm(dl, 'Generating embeddings'):
        embeddings_batch = model(x).detach().cpu()
        embeddings += [embeddings_batch]
    embeddings = torch.cat(embeddings)

    X = embeddings
    y = torch.zeros((X.shape[0],), dtype=torch.long)

    d = Dataset(X, y)
    if kernel_type == 'global':
        d.compute_rbf_kernel(gamma)
    elif kernel_type == 'local':
        d.compute_local_rbf_kernel(gamma)
    else:
        raise KeyError('kernel_type must be either "global" or "local"')
    print('Done.', flush=True)
    class_name = class_name.replace(' ', '_')

    # Prototypes
    if num_prototypes > 0:
        print('Computing prototypes...', end='', flush=True)
        prototype_indices = select_prototypes(d.K, num_prototypes)
        prototypes = torch.stack([ds[i][0] for i in prototype_indices])
        print('Done.', flush=True)
        print(prototype_indices.sort()[0].tolist())

        # Visualize
        if make_plots:
            print('Plotting prototypes...', end='', flush=True)
            num_cols = 8
            num_rows = math.ceil(num_prototypes / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, num_rows * 0.75))
            for i, axis in enumerate(axes.ravel()):
                if i >= num_prototypes:
                    axis.axis('off')
                    continue
                axis.imshow(imagenet_inverse_normalize(prototypes[i]).permute(1,2,0).numpy())
                axis.axis('off')
            fig.suptitle(f'{num_prototypes} Prototypes')
            plt.savefig(output_dir / f'{num_prototypes}_prototypes_imagenet_embeddings_{class_name}.svg')
            print('Done.', flush=True)

        # Criticisms
        if num_criticisms > 0:
            print('Computing criticisms...', end='', flush=True)
            criticism_indices = select_criticisms(d.K, prototype_indices, num_criticisms, regularizer)

            criticisms = torch.stack([ds[i][0] for i in criticism_indices])
            print('Done.', flush=True)
            print(criticism_indices.sort()[0].tolist())

            # Visualize
            if make_plots:
                print('Plotting criticisms...', end='', flush=True)
                num_cols = 8
                num_rows = math.ceil(num_criticisms / num_cols)
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, num_rows * 0.75))
                for i, axis in enumerate(axes.ravel()):
                    if i >= num_criticisms:
                        axis.axis('off')
                        continue
                    axis.imshow(imagenet_inverse_normalize(criticisms[i]).permute(1,2,0).numpy())
                    axis.axis('off')
                fig.suptitle(f'{num_criticisms} Criticisms')
                plt.savefig(output_dir / f'{num_criticisms}_criticisms_imagenet_embeddings_{class_name}.svg')
                print('Done.', flush=True)


else:
    # ====== Run using raw image data, i.e. input images are represented as flattened vector
    print('Preparing data...', end='', flush=True)
    imagenet_tfms_no_normalize = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageNet(root=imagenet_root, split='train', transform=imagenet_tfms_no_normalize)
    class_idx = ds.class_to_idx[class_name]
    ds = torch.utils.data.Subset(ds, torch.where(torch.tensor(ds.targets) == class_idx)[0])
    samples = [sample[0] for sample in ds]

    X = torch.stack(samples).reshape(len(samples), -1)
    y = torch.zeros((X.shape[0],), dtype=torch.long)

    d = Dataset(X, y)
    if kernel_type == 'global':
        d.compute_rbf_kernel(gamma)
    elif kernel_type == 'local':
        d.compute_local_rbf_kernel(gamma)
    else:
        raise KeyError('kernel_type must be either "global" or "local"')
    print('Done.', flush=True)
    class_name = class_name.replace(' ', '_')

    # Prototypes
    if num_prototypes > 0:
        print('Computing prototypes...', end='', flush=True)
        prototype_indices = select_prototypes(d.K, num_prototypes)

        prototypes = d.X[prototype_indices]
        prototype_labels = d.y[prototype_indices]

        sorted_by_y_indices = prototype_labels.argsort()
        prototypes_sorted = prototypes[sorted_by_y_indices]
        prototype_labels = prototype_labels[sorted_by_y_indices]
        print('Done.', flush=True)
        print(prototype_indices.sort()[0].tolist())

        # Visualize
        if make_plots:
            print('Plotting prototypes...', end='', flush=True)
            num_cols = 8
            num_rows = math.ceil(num_prototypes / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, num_rows * 0.75))
            for i, axis in enumerate(axes.ravel()):
                if i >= num_prototypes:
                    axis.axis('off')
                    continue
                axis.imshow(prototypes_sorted[i].view(3,224,224).permute(1,2,0).numpy())
                axis.axis('off')
            fig.suptitle(f'{num_prototypes} Prototypes')
            plt.savefig(output_dir / f'{num_prototypes}_prototypes_imagenet_{class_name}.svg')
            print('Done.', flush=True)

        # Criticisms
        if num_criticisms > 0:
            print('Computing criticisms...', end='', flush=True)
            criticism_indices = select_criticisms(d.K, prototype_indices, num_criticisms, regularizer)

            criticisms = d.X[criticism_indices]
            criticism_labels = d.y[criticism_indices]

            sorted_by_y_indices = criticism_labels.argsort()
            criticisms_sorted = criticisms[sorted_by_y_indices]
            criticism_labels = criticism_labels[sorted_by_y_indices]
            print('Done.', flush=True)
            print(criticism_indices.sort()[0].tolist())

            # Visualize
            if make_plots:
                print('Plotting criticisms...', end='', flush=True)
                num_cols = 8
                num_rows = math.ceil(num_criticisms / num_cols)
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, num_rows * 0.75))
                for i, axis in enumerate(axes.ravel()):
                    if i >= num_criticisms:
                        axis.axis('off')
                        continue
                    axis.imshow(criticisms_sorted[i].view(3,224,224).permute(1,2,0).numpy())
                    axis.axis('off')
                fig.suptitle(f'{num_criticisms} Criticisms')
                plt.savefig(output_dir / f'{num_criticisms}_criticisms_imagenet_{class_name}.svg')
                print('Done.', flush=True)
