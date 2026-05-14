import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_academic_framework():
    # Create the canvas (Clean white background, high resolution proportions)
    fig, ax = plt.subplots(figsize=(16, 7), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 45)
    ax.axis('off')

    # Color Palette - Professional Academic (ICML/NeurIPS style)
    box_ec = '#203A43'      # Dark slate blue for edges
    box_fc = '#F8F9FA'      # Off-white/very light gray for fill
    highlight_fc = '#E8F1F5' # Light blue for important components (RL)
    line_color = '#333333'  # Dark gray for arrows
    text_color = '#1A1A1A'  # Almost black for text
    accent_color = '#D9534F' # Subdued red for error/feedback loops

    # Font setup
    main_font = 'DejaVu Sans' # Safe standard sans-serif

    # Title / Caption
    ax.text(50, 42, "CTNet: Closed-Loop Spatiotemporal BCI Framework", 
            color=text_color, fontsize=18, fontweight='bold', ha='center', va='top', fontname=main_font)

    # Node definitions (x, y, w, h, title, subtitle)
    nodes = {
        'EEG': (5, 18, 12, 8, 'Raw EEG', 'Channels $\\times$ Time'),
        'PREP': (22, 18, 12, 8, 'Preprocessing', 'Bandpass \\& ICA'),
        'CSP': (39, 18, 12, 8, 'OVR-CSP', 'Spatial Filtering'),
        'CNN': (56, 24, 12, 8, '1D-CNN', 'Spatial Features'),
        'LSTM': (56, 12, 12, 8, 'LSTM', 'Temporal Context'),
        'DQN': (75, 14, 14, 14, 'DQN Agent', 'Policy $\\pi(a|s)$\n$Q(s,a; \\theta)$'),
        'ENV': (76.5, 34, 11, 5, 'Robotic Env', 'Kinematics')
    }

    # Helper function for drawing boxes
    def draw_node(key, x, y, w, h, title, sub):
        fc = highlight_fc if key in ['DQN', 'LSTM'] else box_fc
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=box_ec, facecolor=fc, zorder=2)
        
        # Rounded corners approach using FancyBboxPatch for a cleaner look
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.5", 
                                     linewidth=1.5, edgecolor=box_ec, facecolor=fc, zorder=2)
        ax.add_patch(box)
        
        # Text placement
        ax.text(x + w/2, y + h/2 + 1.5, title, color=text_color, fontsize=12, fontweight='bold', ha='center', va='center', fontname=main_font)
        ax.text(x + w/2, y + h/2 - 1.5, sub, color='#4A4A4A', fontsize=10, ha='center', va='center', fontname=main_font, linespacing=1.2)

    # Draw all boxes
    for key, data in nodes.items():
        draw_node(key, *data)

    # Helper function for drawing arrows
    def draw_arrow(x1, y1, x2, y2, color=line_color, ls='-'):
        # Add a slight padding so arrows don't clip into box edges
        pad = 0.5
        ax.annotate('', xy=(x2-pad if x2>x1 else x2+pad, y2-pad if y2>y1 else y2+pad), 
                    xytext=(x1+pad if x1<x2 else x1-pad, y1+pad if y1<y2 else y1-pad),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color=color, ls=ls), zorder=1)

    def draw_orthogonal_arrow(x1, y1, x2, y2, split_x_offset=0.5, color=line_color):
        mid_x = x1 + (x2 - x1) * split_x_offset
        ax.plot([x1, mid_x, mid_x, x2-0.5], [y1, y1, y2, y2], color=color, lw=1.5, zorder=1)
        ax.annotate('', xy=(x2, y2), xytext=(x2-1, y2), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=color), zorder=2)

    # Sequential Connections
    draw_orthogonal_arrow(nodes['EEG'][0]+nodes['EEG'][2], nodes['EEG'][1]+nodes['EEG'][3]/2, 
                          nodes['PREP'][0], nodes['PREP'][1]+nodes['PREP'][3]/2)
                          
    draw_orthogonal_arrow(nodes['PREP'][0]+nodes['PREP'][2], nodes['PREP'][1]+nodes['PREP'][3]/2, 
                          nodes['CSP'][0], nodes['CSP'][1]+nodes['CSP'][3]/2)

    # Branching from CSP to CNN and LSTM
    cx = nodes['CSP'][0] + nodes['CSP'][2]
    cy = nodes['CSP'][1] + nodes['CSP'][3]/2
    
    draw_orthogonal_arrow(cx, cy, nodes['CNN'][0], nodes['CNN'][1]+nodes['CNN'][3]/2, split_x_offset=0.3)
    draw_orthogonal_arrow(cx, cy, nodes['LSTM'][0], nodes['LSTM'][1]+nodes['LSTM'][3]/2, split_x_offset=0.3)

    # Merging from CNN and LSTM to DQN
    rx = nodes['DQN'][0]
    ry = nodes['DQN'][1] + nodes['DQN'][3]/2
    
    draw_orthogonal_arrow(nodes['CNN'][0]+nodes['CNN'][2], nodes['CNN'][1]+nodes['CNN'][3]/2, rx, ry, split_x_offset=0.7)
    draw_orthogonal_arrow(nodes['LSTM'][0]+nodes['LSTM'][2], nodes['LSTM'][1]+nodes['LSTM'][3]/2, rx, ry, split_x_offset=0.7)

    # Feedback loop (DQN -> ENV -> DQN)
    dqn_top_x = nodes['DQN'][0] + nodes['DQN'][2]/2
    dqn_top_y = nodes['DQN'][1] + nodes['DQN'][3]
    env_bot_y = nodes['ENV'][1]
    
    # Action A_t (Up)
    draw_arrow(dqn_top_x, dqn_top_y, dqn_top_x, env_bot_y)
    ax.text(dqn_top_x - 1, dqn_top_y + (env_bot_y - dqn_top_y)/2, "Action $A_t$", color=text_color, fontsize=10, ha='right', va='center')

    # Reward & State (Down)
    env_right_x = nodes['ENV'][0] + nodes['ENV'][2]
    dqn_right_x = nodes['DQN'][0] + nodes['DQN'][2]
    
    route_x = env_right_x + 3
    ax.plot([env_right_x, route_x, route_x], [nodes['ENV'][1]+2.5, nodes['ENV'][1]+2.5, nodes['DQN'][1]+3], color=accent_color, ls='--', lw=1.5, zorder=1)
    ax.annotate('', xy=(dqn_right_x, nodes['DQN'][1]+3), xytext=(route_x, nodes['DQN'][1]+3), 
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=accent_color, ls='--'), zorder=2)
                
    ax.text(route_x + 0.5, nodes['ENV'][1]-2, "Reward $R_t$\nNext State $S_{t+1}$", color=accent_color, fontsize=10, ha='left', va='center')

    # Save outputs
    plt.tight_layout()
    plt.savefig("/home/tomato/A-Brain-Computer-Interface-Control-System-Design-Based-on-Deep-Learning/CTNet/Methodology/ctnet_framework.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig("/home/tomato/A-Brain-Computer-Interface-Control-System-Design-Based-on-Deep-Learning/CTNet/Methodology/ctnet_framework.png", format='png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    create_academic_framework()
