# -*- coding: utf-8 -*-
"""
Azure Speech TTS 批处理脚本（长文分段版）
- 处理 posts/*.txt：第1行=标题；第2行如以 "Date:" 开头则识别日期，否则用今天；其余为正文
- 正文支持以 "Host:" / "Scientist:" 开头的行来切换说话人（不同 voice）
- 自动清理零宽/控制字符；按句分段（句数 & 字符数双阈值）；逐段合成 MP3
- 失败时打印 CancellationDetails；可选用 ffmpeg 合并分段
- 新增参数 --only-full-to-docs：只把 *_full.mp3 复制到 docs/audio
"""

import os, re, html, datetime, pathlib, sys, glob, unicodedata, argparse, subprocess, shutil
from typing import List, Tuple

try:
    import azure.cognitiveservices.speech as speechsdk
except Exception as e:
    print("[ERROR] Missing package 'azure-cognitiveservices-speech':", e)
    sys.exit(1)

DEFAULT_INPUT_GLOB = "posts/*.txt"
DEFAULT_OUT_DIR    = "tts_out"
DEFAULT_MAX_SENTS  = 120
DEFAULT_MAX_CHARS  = 8000
DEFAULT_BREAK_MS   = 250

DEFAULT_OUTPUT_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3

VOICE_HOST_DEFAULT = "en-US-EmmaMultilingualNeural"
VOICE_SCI_DEFAULT  = "en-US-AndrewMultilingualNeural"
RATE_DEFAULT       = "20%"

ROLE_LINE_PAT = re.compile(r'^(host|scientist)\s*:\s*(.+)$', re.I)
SENT_SPLIT = re.compile(r'(?<=[\.\?\!。！？])\s+')

def canonicalize_voice(v: str | None, fallback: str) -> str:
    if not v:
        return fallback
    v = v.strip()
    if ":" in v:
        v = v.split(":", 1)[0].strip()
    if " " in v and not v.endswith("Neural"):
        v = v.split(" ", 1)[0].strip()
    return v or fallback

def sanitize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    for z in ("\u200b", "\u200c",
