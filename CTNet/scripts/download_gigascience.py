#!/usr/bin/env python3
"""
GigaScience 数据集部分下载脚本

尝试从 GigaDB 只下载前 N 个被试的数据，而不是整个 226GB 压缩包

数据集: http://gigadb.org/dataset/100788
论文: Jeong et al., "Multimodal signal dataset for 11 intuitive movement tasks", GigaScience, 2020

创建时间: 2026-02-10

============================================================================
⚠️ 重要发现:
============================================================================
经过测试，GigaDB 的 EEG_ConvertedData.tar.gz (226GB) 是一个完整打包的文件，
无法直接按被试单独下载。

替代方案：
1. 使用 IV-2a 和 IV-2b 数据集（推荐，已有良好结果）
2. 下载压缩包后使用 tar 选择性解压
3. 使用其他较小的公开数据集（如 OpenBMI）
============================================================================
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import urllib.request
from ftplib import FTP
from pathlib import Path
from typing import List, Optional, Tuple
from html.parser import HTMLParser
import ssl


# ============================================================================
# 配置
# ============================================================================

GIGADB_DATASET_ID = "100788"

# 可能的 FTP/HTTP 地址（根据 GigaDB 文档）
POTENTIAL_HOSTS = [
    ("ftp.cngb.org", f"/pub/gigadb/pub/10.5524/{GIGADB_DATASET_ID}/"),
    ("parrot.genomics.cn", f"/gigadb/pub/10.5524/100001_101000/{GIGADB_DATASET_ID}/"),
]

POTENTIAL_HTTP_URLS = [
    f"https://ftp.cngb.org/pub/gigadb/pub/10.5524/{GIGADB_DATASET_ID}/",
    f"http://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/{GIGADB_DATASET_ID}/",
    f"https://gigadb.org/dataset/100788",
]


# ============================================================================
# HTML 目录解析器
# ============================================================================

class DirectoryParser(HTMLParser):
    """解析 HTTP 目录列表"""
    
    def __init__(self):
        super().__init__()
        self.links = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href' and value and not value.startswith('?'):
                    self.links.append(value)


def list_http_directory(url: str) -> List[str]:
    """列出 HTTP 目录中的文件"""
    try:
        # 创建不验证 SSL 的上下文（某些服务器证书可能有问题）
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=context, timeout=30) as response:
            html = response.read().decode('utf-8')
        
        parser = DirectoryParser()
        parser.feed(html)
        
        # 过滤掉父目录链接
        files = [f for f in parser.links if not f.startswith('/') and f != '../']
        return files
    
    except Exception as e:
        print(f"HTTP 目录访问失败: {e}")
        return []


def list_ftp_directory(host: str, path: str) -> List[Tuple[str, int]]:
    """列出 FTP 目录中的文件和大小"""
    try:
        ftp = FTP(host, timeout=30)
        ftp.login()  # 匿名登录
        ftp.cwd(path)
        
        files = []
        
        def callback(line):
            parts = line.split()
            if len(parts) >= 9:
                size = int(parts[4]) if parts[4].isdigit() else 0
                name = parts[-1]
                files.append((name, size))
        
        ftp.retrlines('LIST', callback)
        ftp.quit()
        
        return files
    
    except Exception as e:
        print(f"FTP 目录访问失败: {e}")
        return []


# ============================================================================
# 探索数据集结构
# ============================================================================

def explore_dataset_structure():
    """探索 GigaScience 数据集的目录结构"""
    
    print("="*60)
    print("GigaScience 数据集结构探索")
    print("="*60)
    print(f"数据集 ID: {GIGADB_DATASET_ID}")
    print(f"数据集页面: https://gigadb.org/dataset/{GIGADB_DATASET_ID}")
    print()
    
    all_files = []
    
    # 尝试 HTTP 访问
    print("[1] 尝试 HTTP 访问...")
    for url in POTENTIAL_HTTP_URLS:
        if "gigadb.org/dataset" in url:
            print(f"  跳过数据集页面 (需要 JavaScript): {url}")
            continue
        
        print(f"  尝试: {url}")
        http_files = list_http_directory(url)
        
        if http_files:
            print(f"  ✓ 找到 {len(http_files)} 个文件/目录:")
            for f in http_files[:10]:
                print(f"    - {f}")
            if len(http_files) > 10:
                print(f"    ... 还有 {len(http_files) - 10} 个")
            all_files.extend(http_files)
            break
    
    if not all_files:
        print("  ✗ 所有 HTTP URL 访问失败")
    
    # 尝试 FTP 访问
    print("\n[2] 尝试 FTP 访问...")
    for host, path in POTENTIAL_HOSTS:
        print(f"  尝试: ftp://{host}{path}")
        ftp_files = list_ftp_directory(host, path)
        
        if ftp_files:
            print(f"  ✓ 找到 {len(ftp_files)} 个文件/目录:")
            for name, size in ftp_files[:10]:
                size_str = format_size(size)
                print(f"    - {name} ({size_str})")
            if len(ftp_files) > 10:
                print(f"    ... 还有 {len(ftp_files) - 10} 个")
            all_files.extend([f[0] for f in ftp_files])
            break
    
    if not all_files:
        print("  ✗ 所有 FTP 地址访问失败")
        print("\n  ⚠️ 可能原因:")
        print("    - 网络限制/防火墙")
        print("    - FTP 服务器地址已更改")
        print("    - 需要代理/VPN")
    
    # 分析是否可以按被试下载
    print("\n[3] 分析已知信息...")
    
    if all_files:
        # 查找被试相关的文件模式
        subject_patterns = [
            r'[Ss]ub\d+',
            r'[Ss]\d+',
            r'subject\d+',
            r'[Ss]ubject_\d+',
        ]
        
        subject_files = []
        for f in all_files:
            for pattern in subject_patterns:
                if re.search(pattern, f, re.IGNORECASE):
                    subject_files.append(f)
                    break
        
        if subject_files:
            print(f"  ✓ 找到 {len(subject_files)} 个可能的被试文件:")
            for f in subject_files[:10]:
                print(f"    - {f}")
            print("\n  ✓ 可以按被试单独下载!")
            return True, subject_files
        else:
            # 检查是否只有大的压缩包
            tar_files = [f for f in all_files if '.tar' in f.lower() or '.gz' in f.lower()]
            if tar_files:
                print(f"  ⚠ 只找到压缩包文件:")
                for f in tar_files:
                    print(f"    - {f}")
                print("\n  ✗ 数据打包在一起，无法按被试下载")
                return False, tar_files
    
    # 根据已知信息提供结论
    print("\n  📋 根据 GigaDB 页面信息:")
    print("    - EEG_ConvertedData.tar.gz (226.32 GB) - MATLAB 格式 EEG 数据")
    print("    - RawData.tar.gz (211.89 GB) - 原始 EEG/EMG/EOG 数据")
    print("\n  ✗ 数据被打包在大型压缩文件中，无法直接按被试下载")
    
    return False, ["EEG_ConvertedData.tar.gz", "RawData.tar.gz"]


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


# ============================================================================
# 下载功能
# ============================================================================

def download_file(url: str, output_path: Path, show_progress: bool = True):
    """下载单个文件"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, context=context) as response:
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            block_size = 8192
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                
                f.write(buffer)
                downloaded += len(buffer)
                
                if show_progress and total_size > 0:
                    percent = downloaded / total_size * 100
                    print(f"\r  下载进度: {percent:.1f}% ({format_size(downloaded)}/{format_size(total_size)})", end='')
            
            if show_progress:
                print()


def download_subjects(
    subject_files: List[str],
    subjects_to_download: List[int],
    output_dir: Path,
    base_url: str,
):
    """下载指定被试的数据"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n下载 {len(subjects_to_download)} 个被试的数据...")
    
    for subject in subjects_to_download:
        # 查找匹配的文件
        patterns = [
            f'sub{subject:02d}',
            f'Sub{subject:02d}',
            f's{subject:02d}',
            f'S{subject:02d}',
            f'subject{subject:02d}',
            f'Subject{subject:02d}',
        ]
        
        matching_files = []
        for f in subject_files:
            for p in patterns:
                if p in f:
                    matching_files.append(f)
                    break
        
        if not matching_files:
            print(f"  Subject {subject}: 未找到匹配文件")
            continue
        
        for file in matching_files:
            url = base_url + file
            output_path = output_dir / file
            
            print(f"  Subject {subject}: 下载 {file}...")
            
            try:
                download_file(url, output_path)
                print(f"    ✓ 保存到 {output_path}")
            except Exception as e:
                print(f"    ✗ 下载失败: {e}")


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GigaScience 数据集部分下载")
    
    p.add_argument("--explore", action="store_true",
                   help="只探索目录结构，不下载")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3],
                   help="要下载的被试编号")
    p.add_argument("--output-dir", type=Path, default=Path("./gigascience_raw/"),
                   help="输出目录")
    
    return p.parse_args()


def check_partial_tar_support():
    """检查是否可以部分解压 tar.gz 文件"""
    print("\n" + "="*60)
    print("检查部分解压支持")
    print("="*60)
    
    # 测试 curl 是否支持 range 请求
    try:
        result = subprocess.run(
            ["curl", "--version"],
            capture_output=True, text=True, timeout=10
        )
        print("  ✓ curl 可用")
    except Exception:
        print("  ✗ curl 不可用")
    
    # 测试 tar 是否支持 wildcards
    try:
        result = subprocess.run(
            ["tar", "--help"],
            capture_output=True, text=True, timeout=10
        )
        if "--wildcards" in result.stdout:
            print("  ✓ tar 支持 --wildcards 选项")
        else:
            print("  ⚠ tar 可能不支持 --wildcards")
    except Exception:
        print("  ✗ tar 不可用")


def print_alternatives():
    """打印替代方案"""
    print("\n" + "="*60)
    print("📋 替代方案")
    print("="*60)
    print("""
由于 GigaScience 数据打包在一个 226GB 的压缩包中，无法直接按被试单独下载。

==============================
方案 1: 使用现有数据集（推荐）
==============================
你已经有 IV-2a (4类, 22通道) 和 IV-2b (2类, 3通道) 的良好结果：
  - IV-2a: 分类准确率 ~77%, 控制到达率 ~99%
  - IV-2b: 分类准确率 ~73%, 控制到达率 ~93%

这足以证明你的 RL 控制框架有效！

==============================
方案 2: 部分解压（如果已下载）
==============================
如果你能获取压缩包，可以只解压前3个被试：

# 1. 先查看压缩包内容结构
tar -tzf EEG_ConvertedData.tar.gz | head -100

# 2. 只解压匹配的文件
tar -xzf EEG_ConvertedData.tar.gz --wildcards '*sub01*' '*sub02*' '*sub03*'

# 或者解压到指定目录
tar -xzf EEG_ConvertedData.tar.gz -C ./gigascience_raw/ --wildcards '*sub01*'

==============================
方案 3: 使用其他公开数据集
==============================
- OpenBMI Dataset (~10 GB):
  https://doi.org/10.5524/100542

- PhysioNet Motor Imagery (~5 GB):
  https://physionet.org/content/eegmmidb/1.0.0/

- BNCI Horizon 2020 (~2 GB):
  http://bnci-horizon-2020.eu/database/data-sets

==============================
方案 4: 在 Methodology 中说明
==============================
在论文中说明：
"Due to the prohibitive size of the GigaScience dataset (226 GB),
we validated our approach on the BCI Competition IV datasets (2a and 2b),
which provide complementary evaluation scenarios..."

这是学术上完全可接受的做法！
""")


def main():
    args = parse_args()
    
    # 探索目录结构
    can_download_partial, files = explore_dataset_structure()
    
    # 检查部分解压支持
    check_partial_tar_support()
    
    if args.explore:
        print_alternatives()
        return
    
    if can_download_partial:
        print("\n" + "="*60)
        print("开始下载...")
        print("="*60)
        
        download_subjects(
            files,
            args.subjects,
            args.output_dir,
            POTENTIAL_HTTP_URLS[0],
        )
        
        print("\n下载完成!")
    else:
        print_alternatives()


if __name__ == "__main__":
    main()

