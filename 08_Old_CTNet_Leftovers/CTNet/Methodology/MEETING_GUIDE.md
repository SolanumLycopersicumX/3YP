# 📋 Weekly Meeting Presentation Guide

本指南帮助你准备每周与导师的进度汇报，回应教授的反馈：
> "Communication in project meetings could be clearer. Putting work into a presentation or similar, with a system diagram showing which block is being worked on, and then methods and results for this week would help showcase the work done."

---

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `weekly_progress_template.tex` | Beamer 演示模板（8 页 PPT） |
| `system_diagram.tex` | 独立系统架构图（可单独编译为 PDF/PNG） |
| `methodology_v5.tex` | 完整的方法论文档（详细技术内容） |

---

## 🚀 使用步骤

### 1. 复制模板
```bash
cp weekly_progress_template.tex week_N_progress.tex
```

### 2. 修改高亮模块
在 `week_N_progress.tex` 的系统架构图部分，修改每个模块的 `fill` 颜色：

```latex
% 状态选项:
% completed!40  - 绿色（已完成）
% inprogress!40 - 蓝色（进行中）
% highlight!40  - 红色（本周重点）
% pending!40    - 灰色（待完成）

\node[block, fill=highlight!40] (dqn) {RL Agent\\(DQN)};  % 本周工作
\node[block, fill=completed!40] (prep) {Preprocessing};   % 已完成
```

### 3. 填写内容
每周更新以下部分：
- **Slide 2**: 本周目标（3-5 个具体任务）
- **Slide 3**: 方法和实现细节
- **Slide 4**: 结果（表格 + 图片）
- **Slide 5**: 遇到的挑战和解决方案
- **Slide 6**: 下周计划

### 4. 编译 PDF
```bash
pdflatex week_N_progress.tex
```

---

## 📊 每周汇报清单

### 会议前准备 ✅
- [ ] 更新系统架构图高亮
- [ ] 准备 1-2 个定量结果（准确率/成功率/误差）
- [ ] 准备 1 个可视化（图表/轨迹图/混淆矩阵）
- [ ] 列出 2-3 个问题/需要讨论的点
- [ ] 测试 PDF 是否正确编译

### 汇报结构（5-10 分钟）
1. **架构图** (30s): 指出本周工作的模块
2. **目标** (1min): 本周计划做什么
3. **方法** (2min): 怎么做的
4. **结果** (3min): 做出了什么 + 数据/图表
5. **计划** (1min): 下周做什么
6. **问题** (2min): 需要讨论的点

---

## 🎨 结果展示规范

### 表格格式
```latex
\begin{tabular}{lcc}
    \toprule
    Metric & Baseline & This Week \\
    \midrule
    Accuracy & 72.3\% & \textbf{75.1\%} \\  % 加粗改进
    \bottomrule
\end{tabular}
```

### 插入图片
```latex
\includegraphics[width=0.8\textwidth]{outputs/gym_summary.png}
```

### 代码片段
```latex
\texttt{gym\_control.py} 第 450 行:
\begin{verbatim}
obs, reward, done, info = env.step(action)
\end{verbatim}
```

---

## 📅 建议的周报时间线

| 时间点 | 动作 |
|--------|------|
| 会议前 1 天 | 完成 PPT 初稿 |
| 会议前 2 小时 | 检查图表、编译 PDF |
| 会议后 | 记录反馈到 `logs/` |

---

## 💡 常见问题

### Q: 没有定量结果怎么办？
展示：
- 代码架构图
- 日志/调试输出
- 失败的实验（说明学到了什么）

### Q: 图表太多放不下？
- 主 PPT 放 1-2 个最重要的
- 准备备用 slides 在附录
- 或使用 `\includegraphics[width=0.4\textwidth]` 缩小

### Q: 如何生成系统架构图 PNG？
```bash
pdflatex system_diagram.tex
pdftoppm -png -r 300 system_diagram.pdf system_diagram
```

---

Good luck with your meetings! 🎓






