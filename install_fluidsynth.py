#!/usr/bin/env python3
"""
install_fluidsynth.py — 自动下载 + 解压 FluidSynth Windows 免安装版
=================================================================
来源: GitHub Releases (v2.5.6, win10-x64-glib)
安装: C:/Deepseek/fluidsynth/
"""

import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import os, zipfile, shutil
from pathlib import Path
from urllib.request import urlopen, Request

INSTALL_DIR   = Path("C:/Deepseek/fluidsynth")
FLUIDSYNTH_EXE = INSTALL_DIR / "bin" / "fluidsynth.exe"
GITHUB_URL     = (
    "https://github.com/FluidSynth/fluidsynth/releases/download/"
    "v2.5.6/fluidsynth-v2.5.6-win10-x64-glib.zip"
)
# 备用镜像 (国内访问 GitHub 更快)
MIRROR_URL = (
    "https://ghproxy.net/https://github.com/FluidSynth/fluidsynth/releases/download/"
    "v2.5.6/fluidsynth-v2.5.6-win10-x64-glib.zip"
)

def main():
    print("=" * 60)
    print("FluidSynth Windows 自动安装")
    print("=" * 60)

    # 已安装则跳过
    if FLUIDSYNTH_EXE.exists():
        print(f"\n  已安装: {FLUIDSYNTH_EXE}")
        result = os.popen(f'"{FLUIDSYNTH_EXE}" --version').read()
        print(f"  版本: {result.strip().split(chr(10))[0]}")
        return

    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    # 下载
    zip_path = INSTALL_DIR / "fluidsynth.zip"

    for url in [GITHUB_URL, MIRROR_URL]:
        try:
            print(f"\n  下载: {url}")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunks = []
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  进度: {downloaded/1024:.0f}/{total/1024:.0f} KB ({pct:.0f}%)", end="")
                data = b"".join(chunks)
                print()
            zip_path.write_bytes(data)
            print(f"  完成: {len(data)/1024:.0f} KB")
            break
        except Exception as e:
            print(f"  失败: {e}")
            if url == GITHUB_URL:
                print(f"  尝试镜像...")
                continue
            else:
                print("\n  ❌ 下载失败! 请手动下载:")
                print(f"     {GITHUB_URL}")
                print(f"     解压到: {INSTALL_DIR}")
                return

    # 解压
    print(f"\n  解压到: {INSTALL_DIR}")
    with zipfile.ZipFile(zip_path) as zf:
        # 获取顶层目录名
        root = zf.namelist()[0].split("/")[0]
        zf.extractall(INSTALL_DIR)

    # 如果解压到子目录, 移到上层
    extracted = INSTALL_DIR / root
    if extracted.is_dir() and extracted != INSTALL_DIR:
        for item in extracted.iterdir():
            dest = INSTALL_DIR / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        extracted.rmdir()

    # 清理
    zip_path.unlink()

    # 验证
    if FLUIDSYNTH_EXE.exists():
        result = os.popen(f'"{FLUIDSYNTH_EXE}" --version').read()
        print(f"\n  ✅ 安装完成!")
        print(f"  路径: {FLUIDSYNTH_EXE}")
        print(f"  版本: {result.strip().split(chr(10))[0]}")
    else:
        print(f"\n  ⚠️  解压完成但未找到 {FLUIDSYNTH_EXE}")
        print(f"  目录内容:")
        for f in sorted(INSTALL_DIR.rglob("*")):
            print(f"    {f.relative_to(INSTALL_DIR)}")

    print(f"\n  现在可以运行:")
    print(f"    python guofeng_data_factory.py")


if __name__ == "__main__":
    main()
