import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np

def create_framework():
    # Create the canvas
    fig, ax = plt.subplots(figsize=(24, 12), facecolor='#111111')
    ax.set_facecolor('#111111')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # Color Palette - Brutalist minimalism
    box_color = '#111111'
    line_color = '#cccccc'
    accent_color = '#00ffcc' # Cyberpunk cyan accent
    text_color = '#f0f0f0'
    faded_text = '#777777'

    # Setup typography
    ax.text(3, 46, "CTNet PIPELINE", color=text_color, fontsize=32, fontweight='bold', ha='left', va='top', fontname='sans-serif')
    ax.text(3, 44, "NEURAL FORMALISM / SPATIOTEMPORAL SCHEMATIC", color=accent_color, fontsize=12, fontweight='bold', ha='left', va='top', fontname='sans-serif')
    
    ax.text(97, 46, "FIG. 01", color=faded_text, fontsize=10, ha='right', va='top', fontname='sans-serif', rotation=0)
    
    # Background grid for mathematical precision
    for x in range(0, 100, 2):
        ax.plot([x, x], [0, 50], color='#1a1a1a', lw=1, zorder=0)
    for y in range(0, 50, 2):
        ax.plot([0, 100], [y, y], color='#1a1a1a', lw=1, zorder=0)

    # Nodes (x, y, w, h, title, sub)
    nodes = {
        'EEG': (5, 20, 12, 10, 'RAW EEG', 'ORGANIC CORTICAL\nSIGNAL ACQUISITION'),
        'PREP': (22, 20, 12, 10, 'FILTER & ICA', 'MNE / 8-30Hz / ARTIFACT\nREJECTION MATRIX'),
        'CSP': (39, 20, 12, 10, 'OVR-CSP', 'SPATIAL FEATURE\nEXTRACTION PROJECTIONS'),
        'CNN': (56, 26, 12, 8, '1D-CNN', 'LOCAL SENSORIMOTOR\nDYNAMICS CAPTURE'),
        'LSTM': (56, 12, 12, 8, 'LSTM MEMORY', 'TEMPORAL DRIFT\nMITIGATION CONTEXT'),
        'RL': (75, 17, 14, 16, 'DQN POLICY', 'Markov Decision Process\nQ(s, a) OPTIMIZATION'),
        'SYS': (76.5, 40, 11, 4, 'ROBOTIC ACTUATION', 'PHYSICAL KINEMATICS')
    }

    # Edges
    connections = [('EEG', 'PREP'), ('PREP', 'CSP')]
    
    def draw_path(x1, y1, x2, y2, color=line_color):
        dx = (x2 - x1) / 2
        ax.plot([x1, x1+dx, x1+dx, x2], [y1, y1, y2, y2], color=color, lw=1.5, zorder=1)

    for src, dst in connections:
        sx = nodes[src][0] + nodes[src][2]
        sy = nodes[src][1] + nodes[src][3]/2
        dx = nodes[dst][0]
        dy = nodes[dst][1] + nodes[dst][3]/2
        draw_path(sx, sy, dx, dy)
        ax.scatter([sx + (dx - sx)/2], [sy], color=accent_color, s=20, zorder=3)

    # Split to CNN / LSTM
    cx = nodes['CSP'][0] + nodes['CSP'][2]
    cy = nodes['CSP'][1] + nodes['CSP'][3]/2
    
    for node in ['CNN', 'LSTM']:
        nx = nodes[node][0]
        ny = nodes[node][1] + nodes[node][3]/2
        draw_path(cx, cy, nx, ny)

    # Merge to RL
    rx = nodes['RL'][0]
    ry = nodes['RL'][1] + nodes['RL'][3]/2
    for node in ['CNN', 'LSTM']:
        sx = nodes[node][0] + nodes[node][2]
        sy = nodes[node][1] + nodes[node][3]/2
        draw_path(sx, sy, rx, ry)

    # Env Feedback 
    sys_x = nodes['SYS'][0] + nodes['SYS'][2]/2
    sys_y = nodes['SYS'][1]
    rl_x = nodes['RL'][0] + nodes['RL'][2]/2
    rl_y = nodes['RL'][1] + nodes['RL'][3]
    
    # Action (Up)
    ax.annotate('', xy=(sys_x, sys_y), xytext=(rl_x, rl_y),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=accent_color), zorder=2)
                
    # Reward (Down, dashed)
    sx_out = nodes['SYS'][0] + nodes['SYS'][2]
    rl_out = nodes['RL'][0] + nodes['RL'][2]
    ax.plot([sx_out+2, sx_out+2], [nodes['SYS'][1]+2, rl_y-2], color=accent_color, ls='--', lw=1.5)
    ax.plot([sx_out, sx_out+2], [nodes['SYS'][1]+2, nodes['SYS'][1]+2], color=accent_color, ls='--', lw=1.5)
    ax.annotate('', xy=(rl_out, rl_y-2), xytext=(sx_out+2, rl_y-2),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=accent_color, ls='--'), zorder=2)

    ax.text(rl_x - 1, rl_y + 1, "ACTION A_t", color=accent_color, fontsize=9, va='bottom', ha='right', fontname='sans-serif')
    ax.text(sx_out + 2.5, nodes['SYS'][1]+2, "REWARD R_t\nSTATE S_{t+1}", color=accent_color, fontsize=9, va='center', ha='left', fontname='sans-serif')

    # Draw Boxes
    for key, (x, y, w, h, title, sub) in nodes.items():
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=line_color, facecolor=box_color, zorder=2)
        ax.add_patch(rect)
        
        # Cyber-accent rect
        if key in ['RL', 'LSTM']:
            accent_rect = patches.Rectangle((x, y), 0.6, h, linewidth=0, facecolor=accent_color, zorder=3)
            ax.add_patch(accent_rect)
            
        ax.text(x + 1.5, y + h - 2, title, color=text_color, fontsize=14, fontweight='bold', fontname='sans-serif')
        ax.text(x + 1.5, y + h - 5.5, sub, color=faded_text, fontsize=9, fontname='sans-serif', va='bottom', linespacing=1.5)
        
        # Museum precision crosshairs
        c = 0.8
        for px, py in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
            ax.plot([px-c, px+c], [py, py], color=faded_text, lw=0.5, zorder=4)
            ax.plot([px, px], [py-c, py+c], color=faded_text, lw=0.5, zorder=4)

    # Decorative telemetry
    ax.text(5, 5, "OBSERVATIONAL TELEMETRY // SYSTEM STATE", color=faded_text, fontsize=8, fontname='sans-serif')
    for i in range(120):
        val = np.abs(np.sin(i * 0.1) * np.exp(-i * 0.02) + np.random.normal(0, 0.2))
        h = 2 + val * 4
        c = accent_color if i % 15 == 0 else line_color
        ax.plot([5 + i*0.3, 5 + i*0.3], [2, h], color=c, lw=1)

    plt.tight_layout()
    plt.savefig("/home/tomato/A-Brain-Computer-Interface-Control-System-Design-Based-on-Deep-Learning/CTNet/Methodology/ctnet_framework.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig("/home/tomato/A-Brain-Computer-Interface-Control-System-Design-Based-on-Deep-Learning/CTNet/Methodology/ctnet_framework.png", dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor(), edgecolor='none')

if __name__ == "__main__":
    create_framework()
