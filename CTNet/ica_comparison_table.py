#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate ICA Effect Comparison Table
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Original\n(mymat_raw)', 'Bandpass Filter\n(8-30Hz)', 'Bandpass +\nFastICA']
accuracy = [79.40, 68.67, 66.13]
kappa = [72.53, 58.35, 54.84]
colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Accuracy Comparison
ax1 = axes[0]
bars1 = ax1.bar(methods, accuracy, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_title('Classification Accuracy Comparison', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylim(0, 100)
ax1.axhline(y=79.40, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=2)
ax1.text(2.3, 80, 'Baseline: 79.40%', fontsize=10, color='#2ecc71', fontweight='bold')

# Add value labels
for bar, acc in zip(bars1, accuracy):
    height = bar.get_height()
    ax1.annotate(f'{acc:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=14, fontweight='bold')

# Plot 2: Change from baseline
ax2 = axes[1]
changes = [0, accuracy[1] - accuracy[0], accuracy[2] - accuracy[0]]
bar_colors = ['#2ecc71', '#e74c3c', '#e74c3c']
bars2 = ax2.bar(methods, changes, color=bar_colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Change from Baseline (%)', fontsize=14, fontweight='bold')
ax2.set_title('Effect of Preprocessing on Accuracy', fontsize=16, fontweight='bold', pad=15)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_ylim(-20, 5)

# Add value labels
for bar, change in zip(bars2, changes):
    height = bar.get_height()
    label = f'{change:+.2f}%' if change != 0 else 'Baseline'
    y_pos = height + 0.5 if height >= 0 else height - 1.5
    ax2.annotate(label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5 if height >= 0 else -15),
                textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=14, fontweight='bold',
                color='#27ae60' if change == 0 else '#c0392b')

plt.tight_layout()

# Add summary text box
fig.text(0.5, -0.03, 
         'Conclusion: Additional preprocessing (bandpass filtering and ICA) does NOT improve CTNet performance.\n'
         'The original mymat_raw data achieves the best accuracy (79.40%).',
         ha='center', fontsize=12, style='italic',
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'))

plt.savefig('ica_comparison_chart.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Chart saved: ica_comparison_chart.png")

# Also create a detailed table image
fig2, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Table data
table_data = [
    ['Preprocessing Method', 'Accuracy (%)', 'Kappa', 'Change', 'Recommendation'],
    ['Original (mymat_raw)', '79.40', '72.53', 'Baseline', '✓ RECOMMENDED'],
    ['Bandpass Filter (8-30Hz)', '68.67', '58.35', '-10.73%', '✗ Not recommended'],
    ['Bandpass + FastICA', '66.13', '54.84', '-13.27%', '✗ Not recommended'],
]

# Create table
table = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc='center',
    loc='center',
    colWidths=[0.28, 0.15, 0.12, 0.15, 0.20]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.5)

# Header style
for j in range(5):
    table[(0, j)].set_facecolor('#34495e')
    table[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=13)

# Row colors
row_colors = ['#d5f5e3', '#fdebd0', '#fadbd8']  # Green, Orange, Red tints
for i in range(1, 4):
    for j in range(5):
        table[(i, j)].set_facecolor(row_colors[i-1])

# Highlight recommendation column
table[(1, 4)].set_text_props(color='#27ae60', fontweight='bold')
table[(2, 4)].set_text_props(color='#e74c3c', fontweight='bold')
table[(3, 4)].set_text_props(color='#e74c3c', fontweight='bold')

# Title
ax.set_title('ICA Preprocessing Effect on CTNet Classification Performance\n(BCI Competition IV-2a Dataset)',
             fontsize=16, fontweight='bold', pad=20)

# Add notes
notes = """
Key Findings:
• Original mymat_raw data achieves the best performance (79.40% accuracy)
• Additional bandpass filtering (8-30Hz) reduces accuracy by 10.73%
• Adding FastICA further reduces accuracy by 2.54% (total -13.27% from baseline)
• Conclusion: For CTNet training, use the original mymat_raw data directly without additional ICA preprocessing
"""
ax.text(0.5, -0.15, notes, transform=ax.transAxes, fontsize=11,
        ha='center', va='top', style='italic',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6', pad=0.5))

plt.tight_layout()
plt.savefig('ica_comparison_table.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Table saved: ica_comparison_table.png")

plt.show()

