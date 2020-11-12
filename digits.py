import math
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_svmlight_file
import torch

from mmd_critic import Dataset, select_prototypes, select_criticisms


cwd = Path('.')
data_dir = cwd / 'data'
output_dir = cwd / 'output'

data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

gamma = 0.026

num_prototypes = 32
num_criticisms = 10

kernel_type = 'local'
# kernel_type = 'global'

# regularizer = None
regularizer = 'logdet'
# regularizer = 'iterative'

make_plots = True

print('==============')
print(f'data_dir:{data_dir.absolute()}')
print(f'output_dir:{output_dir.absolute()}')
print(f'num_prototypes:{num_prototypes}')
print(f'num_criticisms:{num_criticisms}')
print(f'gamma:{gamma}')
print(f'kernel_type:{kernel_type}')
print(f'regularizer:{regularizer}')
print(f'make_plots:{make_plots}')
print('==============\n')

# torch.set_num_threads(64)

# Data setup
def load_data(fname):
    X, y = load_svmlight_file(str(data_dir / fname))
    X = torch.tensor(X.todense(), dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    sort_indices = y.argsort() # torch.argsort does not match np.argsort
    # sort_indices = y.numpy().argsort() # argsort is not stable if quicksort is used
    # print(sort_indices)
    X = X[sort_indices, :]
    y = y[sort_indices]
    return X, y


print('Preparing data...', end='', flush=True)
X_train, y_train = load_data('usps')
X_test, y_test = load_data('usps.t')
y_train -= 1
y_test -= 1

d_train = Dataset(X_train, y_train)
if kernel_type == 'global':
    d_train.compute_rbf_kernel(gamma)
elif kernel_type == 'local':
    d_train.compute_local_rbf_kernel(gamma)
else:
    raise KeyError('kernel_type must be either "global" or "local"')
print('Done.', flush=True)

# Prototypes
if num_prototypes > 0:
    print('Computing prototypes...', end='', flush=True)
    prototype_indices = select_prototypes(d_train.K, num_prototypes)

    prototypes = d_train.X[prototype_indices]
    prototype_labels = d_train.y[prototype_indices]

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
            axis.imshow(prototypes_sorted[i].view(16,16).numpy(), cmap='gray')
            axis.axis('off')
        fig.suptitle(f'{num_prototypes} Prototypes')
        plt.savefig(output_dir / f'{num_prototypes}_prototypes_digits.svg')
        print('Done.', flush=True)

    # Criticisms
    if num_criticisms > 0:
        print('Computing criticisms...', end='', flush=True)
        criticism_indices = select_criticisms(d_train.K, prototype_indices, num_criticisms, regularizer)

        criticisms = d_train.X[criticism_indices]
        criticism_labels = d_train.y[criticism_indices]

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
                axis.imshow(criticisms_sorted[i].view(16,16).numpy(), cmap='gray')
                axis.axis('off')
            fig.suptitle(f'{num_criticisms} Criticisms')
            plt.savefig(output_dir / f'{num_criticisms}_criticisms_digits.svg')
            print('Done.', flush=True)

