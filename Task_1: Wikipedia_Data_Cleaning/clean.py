import argparse
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import requests
import ujson as uj
from mwparserfromhell import parse as mw_parse
from tqdm import tqdm


def download(url: str, dest: Path, chunk_size: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[OK] 已存在 {dest}, 跳过下载")
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading dump") as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"[OK] 下载完成: {dest}")


def run_wikiextractor(dump_path: Path, out_dir: Path, processes: int):
    try:
        import wikiextractor  # noqa: F401
    except ModuleNotFoundError:
        print("[WARN] wikiextractor 未安装，自动 pip install …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wikiextractor"])

    out_dir.mkdir(parents=True, exist_ok=True)

    common_opts = ["--json", "--processes", str(processes), "--bytes", "4096K", "-o", str(out_dir), str(dump_path)]

    variants = [
        [sys.executable, "-m", "wikiextractor"],
        [sys.executable, "-m", "wikiextractor.WikiExtractor"],
        ["wikiextractor"],
    ]

    for v in variants:
        cmd = v + common_opts
        print(f"[ ] 尝试: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            print("[OK] WikiExtractor 完成")
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[WARN] 方案失败: {e}")
    raise RuntimeError("无法调用 wikiextractor，请检查 PATH 或 pip 包版本。")

RE_TEMPLATE = re.compile(r"\{\{.*?\}\}")
RE_REF = re.compile(r"<ref.*?</ref>")
RE_PARENTHESES = re.compile(r"[（(].*?[）)]")
RE_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = RE_TEMPLATE.sub("", text)
    text = RE_REF.sub("", text)
    text = RE_PARENTHESES.sub("", text)
    text = RE_SPACES.sub(" ", text)
    return text.strip()


def is_keep(text: str, min_len: int = 100, en_ratio_th: float = 0.4) -> bool:
    if len(text) < min_len:
        return False
    en_chars = sum(c.isascii() and c.isalpha() for c in text)
    return en_chars / len(text) < en_ratio_th


def process_file(fp: Path):
    results = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            item = uj.loads(line)
            raw_txt = item.get("text", "")
            if len(raw_txt) < 100:
                continue
            plain = mw_parse(raw_txt).strip_code()
            plain = clean_text(plain)
            if not is_keep(plain):
                continue
            results.append(uj.dumps({"text": plain, "meta": {k: item.get(k) for k in ("id", "title", "url")}}, ensure_ascii=False))
    return results


def clean_and_convert(extracted_dir: Path, out_jsonl: Path, workers: int):
    files = list(extracted_dir.rglob("wiki_*"))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with mp.Pool(workers) as pool, open(out_jsonl, "w", encoding="utf-8") as fout, tqdm(total=len(files), desc="Cleaning & converting", unit="file") as pbar:
        for lines in pool.imap_unordered(process_file, files):
            if lines:
                fout.write("\n".join(lines) + "\n")
            pbar.update(1)
    print(f"[OK] 清洗并写出 {out_jsonl}")


def sample_jsonl(src: Path, dest: Path, n: int):
    with open(src, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for line in tqdm(lines[:n], desc="Sampling", unit="line"):
            f.write(line)
    print(f"[OK] 采样 {n} 条 -> {dest}")


def parse_args():
    p = argparse.ArgumentParser(description="中文维基百科 dump 清洗管道")
    p.add_argument("--url", default="https://dumps.wikimedia.org/zhwiki/20250201/zhwiki-20250201-pages-articles-multistream.xml.bz2")
    p.add_argument("--work_dir", default="./data")
    p.add_argument("--sample_size", type=int, default=1000)
    p.add_argument("--processes", type=int, default=os.cpu_count() or 4)
    return p.parse_args()


def main():
    args = parse_args()
    work = Path(args.work_dir).resolve()
    raw_path = work / "raw/zhwiki.xml.bz2"
    extracted_dir = work / "extracted"
    clean_jsonl = work / "clean/zhwiki_clean.jsonl"
    sample_jsonl_path = work / f"clean/zhwiki_sample_{args.sample_size}.jsonl"

    steps = ["下载 dump", "抽取文本", "清洗转换", "抽样"]
    with tqdm(total=len(steps), desc="Pipeline", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {desc}") as stepbar:
        download(args.url, raw_path)
        stepbar.update(); stepbar.set_description(steps[0])

        run_wikiextractor(raw_path, extracted_dir, processes=args.processes)
        stepbar.update(); stepbar.set_description(steps[1])

        clean_and_convert(extracted_dir, clean_jsonl, workers=args.processes)
        stepbar.update(); stepbar.set_description(steps[2])

        sample_jsonl(clean_jsonl, sample_jsonl_path, n=args.sample_size)
        stepbar.update(); stepbar.set_description(steps[3])

    print("全部流程结束，结果文件：")
    print(f"  - 清洗后：{clean_jsonl}")
    print(f"  - 抽样：  {sample_jsonl_path}")


if __name__ == "__main__":
    main()

