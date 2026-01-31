import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define base directory
base_dir = Path(__file__).parent

# Define dataset categories and their subdirectories
categories = ['auto_test', 'test', 'train', 'val']
subdirs = [0, 1, 2, 3, 4]

# Count images in each subdirectory
data = {category: [] for category in categories}

for category in categories:
    for subdir in subdirs:
        path = base_dir / category / str(subdir)
        if path.exists():
            # Count image files (common extensions)
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
            count = sum(1 for f in path.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions)
            data[category].append(count)
        else:
            data[category].append(0)

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(subdirs))
width = 0.2

for i, category in enumerate(categories):
    offset = width * (i - 1.5)
    ax.bar(x + offset, data[category], width, label=category.replace('_', ' ').title())
    ax.bar_label(ax.containers[i], label_type='edge')

ax.set_xlabel('Subdirectory (0-4)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax.set_title('Number of Images per Subdirectory', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subdirs)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(base_dir / 'images_per_subdirectory.png', dpi=300, bbox_inches='tight')
plt.show()
