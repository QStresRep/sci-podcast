# -*- coding: utf-8 -*-
"""
Azure Speech TTS 批处理脚本（稳健版：自动切分 + 指数退避 + 合并重编码）
- 输入：posts/*.txt
  * 第1行: 标题；第2行: 可选 "Date: YYYY-MM-DD"；其余为正文
  * 正文可用 "Host:" / "Scientist:" 标注角色（不同 voice）
- 防护：
  * 每段生成前检测 SSML 长度，>4500 自动再切分
  * 对每段合成加入指数退避重试（默认最多5次）
  * 失败时打印 Azure 真实错误详情（CancellationDetails）
  * 可选：--silence-on-fail 失败段落写 1s 静音占位，保证最终能合并
- 合并：
  * 始终使用 concat list + re-encode，避免参数不一致导致拼接丢段
- 发布：
  * --only-full-to-docs 仅复制 *_full.mp3 到 docs/audio
"""

import os, re, html, datetime, pathlib, sys, glob, unicodedata, argparse, subprocess, time, random
from typing import List, Tuple

try:
    import azure.cognitiveservices.speech as speechsdk
except Exception as e:
    print("[ERROR] Missing package 'azure-cognitiveservices-speech':", e)
    sys.exit(1)

# ---------- 默认参数 ----------
DEFAULT_INPUT_GLOB = "posts/*.txt"
DEFAULT_OUT_DIR    = "tts_out"
DEFAULT_MAX_SENTS  = 20       # 更保守
DEFAULT_MAX_CHARS  = 900      # 更保守
DEFAULT_BREAK_MS   = 250

DEFAULT_OUTPUT_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3

VOICE_HOST_DEFAULT = "en-US-JennyNeural"    # 建议用广域可用的标准神经音色
VOICE_SCI_DEFAULT  = "en-US-GuyNeural"
RATE_DEFAULT       = "30%"

ROLE_LINE_PAT = re.compile(r'^(host|scientist)\s*:\s*(.+)$', re.I)
SENT_SPLIT = re.compile(r'(?<=[\.\?\!。！？])\s+')

SSML_MAX_LEN = 4500  # 安全阈值（含标签）

# ---------- 工具函数 ----------
def canonicalize_voice(v: str | None, fallback: str) -> str:
    """把奇怪后缀统一成稳定可用的 Neural 名称；清理空格/花样标识。"""
    if not v:
        return fallback
    v = v.strip()
    if ":" in v:
        v = v.split(":", 1)[0].strip()
    # 把 MultilingualNeural 等映射到 Neural，避免区域不可用
    v = v.replace("MultilingualNeural", "Neural")
    # 去掉市场名/花样标识
    for junk in ["Dragon", "HD", "Latest"]:
        v = v.replace(junk, "")
    v = re.sub(r"\s+", "", v)
    return v or fallback

def sanitize_text(s: str) -> str:
    """基本清洗：换行统一、去零宽/控制符、NFC 规范化。"""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    for z in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"):
        s = s.replace(z, "")
    s = "".join(ch for ch in s if (ch >= " " or ch in "\n\t"))
    return unicodedata.normalize("NFC", s)

def strict_clean_sentence(s: str) -> str:
    """更严格的清洗：去掉冷门/可能触发解析的字符（可用 --strict-clean 开启）。"""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}%/:+&=_"
    return "".join(ch if ch in allowed else " " for ch in s)

def slugify(s: str) -> str:
    s = s.strip().lower().replace(" ", "-")
    return re.sub(r"[^a-z0-9_-]+", "-", s)[:80].strip("-") or "episode"

def parse_role_line(line: str, voice_host: str, voice_sci: str) -> Tuple[str, str]:
    m = ROLE_LINE_PAT.match(line)
    if m:
        role = m.group(1).lower()
        content = m.group(2).strip()
        voice = voice_host if role == "host" else voice_sci
        return voice, content
    return voice_host, line.strip()

def to_sentences(text: str, strict: bool = False):
    segs = [seg.strip() for seg in SENT_SPLIT.split(text) if seg and seg.strip()]
    if strict:
        segs = [strict_clean_sentence(x) for x in segs]
    return segs

def build_dialog_items(body: str, voice_host: str, voice_sci: str, strict: bool):
    items = []
    for raw_ln in body.split("\n"):
        ln = raw_ln.strip()
        if not ln:
            continue
        voice, content = parse_role_line(ln, voice_host, voice_sci)
        sents = to_sentences(content, strict=strict) or [content]
        items.append((voice, sents))
    return items

def chunk_dialog_items(items, max_sents: int, max_chars: int):
    chunks, cur = [], []
    sent_count = 0
    char_count = 0

    def flush():
        nonlocal cur, sent_count, char_count
        if cur:
            chunks.append(cur)
            cur, sent_count, char_count = [], 0, 0

    for voice, sents in items:
        for s in sents:
            s_len = len(s)
            if (sent_count + 1 > max_sents) or (char_count + s_len > max_chars):
                flush()
            if cur and cur[-1][0] == voice:
                cur[-1][1].append(s)
            else:
                cur.append((voice, [s]))
            sent_count += 1
            char_count += s_len
    flush()
    return chunks

def build_ssml_from_chunk(chunk, rate: str, break_ms: int) -> str:
    parts = []
    for voice, sents in chunk:
        inner = "".join(
            "<s>" + html.escape(seg) + "</s><break time='" + str(break_ms) + "ms'/>"
            for seg in sents
        )
        parts.append(
            '<voice name="' + voice + '"><prosody rate="' + html.escape(rate) + '">' + inner + "</prosody></voice>"
        )
    return '<speak version="1.0" xml:lang="en-US">' + "".join(parts) + "</speak>"

def split_chunk_in_half(chunk):
    if len(chunk) <= 1:
        return [chunk]
    mid = len(chunk) // 2
    return [chunk[:mid], chunk[mid:]]

# ---------- Azure 合成（带退避重试） ----------
def synth_ssml_with_retry(ssml: str, out_path: str, prefer_voice_for_config: str, output_format,
                          retries: int = 5, base_delay: float = 0.6, jitter: float = 0.4):
    key = os.getenv("SPEECH_KEY"); region = os.getenv("SPEECH_REGION")
    if not key or not region:
        raise SystemExit("Missing SPEECH_KEY / SPEECH_REGION secrets.")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = prefer_voice_for_config
    speech_config.set_speech_synthesis_output_format(output_format)
    try:
        speech_config.set_profanity(speechsdk.ProfanityOption.Removed)
    except Exception:
        pass

    audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    if len(ssml) > SSML_MAX_LEN:
        raise RuntimeError(f"SSML too long ({len(ssml)}), need split")

    attempt = 0
    while True:
        attempt += 1
        print(f"[DEBUG] synth attempt={attempt} ssml_len={len(ssml)} -> {out_path}")
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return  # success

        # 解析错误详情
        try:
            details = speechsdk.CancellationDetails.from_result(result)
            err = f"reason={details.reason}; error={details.error_details}"
        except Exception:
            err = f"reason={result.reason}"

        # 可重试类：限流/瞬断/未知取消 → 退避
        retriable = any(k in (err or "").lower() for k in [
            "timeout", "throttle", "429", "temporarily", "connection", "network", "quota", "rate limit", "server busy"
        ])

        if attempt <= retries and retriable:
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            print(f"[WARN] TTS canceled ({err}). retrying in {delay:.2f}s ...")
            time.sleep(delay)
            continue

        # 不可重试或到达重试上限
        raise RuntimeError(f"TTS canceled: {err}")

# ---------- 文件处理 ----------
def process_file(txt_path: pathlib.Path, out_dir: pathlib.Path, voice_host: str, voice_sci: str,
                 rate: str, max_sents: int, max_chars: int, break_ms: int, output_format,
                 strict_clean: bool, silence_on_fail: bool, retries: int, base_delay: float, jitter: float):
    raw = txt_path.read_text(encoding="utf-8").splitlines()
    if len(raw) < 3:
        print("[WARN] too short:", txt_path)
        return []

    title = (raw[0] or "Episode").strip()
    date = None
    if raw[1].lower().startswith("date:"):
        date = raw[1].split(":", 1)[1].strip()
    if not date:
        date = datetime.date.today().isoformat()

    body = sanitize_text("\n".join(raw[2:]).strip())
    if len(body) < 20:
        print("[WARN] body short:", txt_path)
        return []

    safe_date = date.replace("-", "")
    base_name = safe_date + "_" + slugify(title) + ".mp3"
    base_out = out_dir / base_name

    items  = build_dialog_items(body, voice_host, voice_sci, strict=strict_clean)
    chunks = chunk_dialog_items(items, max_sents=max_sents, max_chars=max_chars)
    outputs: List[pathlib.Path] = []

    print("[INFO] chunks=" + str(len(chunks)))

    for idx, chunk in enumerate(chunks, 1):
        subs = [chunk]
        # 预检查并按需切分，直到所有子块 SSML 不超阈值
        while True:
            need_split = False
            new_subs = []
            for c in subs:
                ssml = build_ssml_from_chunk(c, rate, break_ms)
                if len(ssml) > SSML_MAX_LEN:
                    need_split = True
                    new_subs.extend(split_chunk_in_half(c))
                else:
                    new_subs.append(c)
            subs = new_subs
            if not need_split:
                break

        for j, sub in enumerate(subs, 1):
            ssml_sub = build_ssml_from_chunk(sub, rate, break_ms)
            out_path = base_out if (len(chunks) == 1 and len(subs) == 1) else \
                base_out.with_name(base_out.stem + f"_part{idx}_{j}" + base_out.suffix)
            print("[TTS]", txt_path, "->", out_path)

            try:
                synth_ssml_with_retry(
                    ssml_sub, str(out_path),
                    prefer_voice_for_config=voice_host,
                    output_format=output_format,
                    retries=retries, base_delay=base_delay, jitter=jitter
                )
                print("[OK] wrote", out_path, "(", out_path.stat().st_size, "bytes )")
                outputs.append(out_path)
            except Exception as e:
                print("[FAIL segment]", out_path, ":", e)
                if silence_on_fail:
                    # 写 1 秒静音占位，保证合并不中断
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-f", "lavfi",
                            "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
                            "-t", "1", "-q:a", "9", str(out_path)
                        ], check=True)
                        print("[OK] wrote silence placeholder ->", out_path)
                        outputs.append(out_path)
                    except Exception as ee:
                        print("[FAIL] write silence failed:", ee)

    return outputs

# ---------- 合并 ----------
def merge_parts_with_ffmpeg(parts: List[pathlib.Path], merged_path: pathlib.Path):
    if not parts:
        return False
    try:
        lst = merged_path.with_suffix(".txt")
        lines = ["file '" + str(p).replace("'", "'\\''") + "'" for p in parts]
        lst.write_text("\n".join(lines), encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c:a", "libmp3lame", "-b:a", "160k", str(merged_path)],
            check=True
        )
        lst.unlink(missing_ok=True)
        print("[OK] merged (re-encoded) ->", merged_path)
        return True
    except Exception as e:
        print("[FAIL] merge failed:", e)
        return False

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser(description="Azure Speech TTS batch (robust).")
    ap.add_argument("--input-glob", default=os.getenv("INPUT_GLOB", DEFAULT_INPUT_GLOB))
    ap.add_argument("--out-dir",    default=os.getenv("OUT_DIR", DEFAULT_OUT_DIR))
    ap.add_argument("--max-sents",  type=int, default=int(os.getenv("MAX_SENTS", DEFAULT_MAX_SENTS)))
    ap.add_argument("--max-chars",  type=int, default=int(os.getenv("MAX_CHARS", DEFAULT_MAX_CHARS)))
    ap.add_argument("--break-ms",   type=int, default=int(os.getenv("BREAK_MS", DEFAULT_BREAK_MS)))
    ap.add_argument("--merge",      action="store_true", help="Merge chunked parts into a single MP3 (requires ffmpeg).")
    ap.add_argument("--voice-host", default=canonicalize_voice(os.getenv("VOICE_HOST"), VOICE_HOST_DEFAULT))
    ap.add_argument("--voice-sci",  default=canonicalize_voice(os.getenv("VOICE_SCI"),  VOICE_SCI_DEFAULT))
    ap.add_argument("--rate",       default=os.getenv("SPEED", RATE_DEFAULT))
    ap.add_argument("--use-48k",    action="store_true", help="Use 48k/192k MP3 output instead of default 24k/160k.")
    ap.add_argument("--only-full-to-docs", action="store_true",
                    help="Copy only *_full.mp3 into docs/audio (for publishing).")
    ap.add_argument("--strict-clean", action="store_true",
                    help="Apply strict character filtering on sentences.")
    ap.add_argument("--silence-on-fail", action="store_true",
                    help="Write 1s silence for a failed segment, instead of aborting the whole run.")
    ap.add_argument("--retries", type=int, default=int(os.getenv("TTS_RETRIES", 5)))
    ap.add_argument("--retry-base", type=float, default=float(os.getenv("TTS_RETRY_BASE", 0.6)))
    ap.add_argument("--retry-jitter", type=float, default=float(os.getenv("TTS_RETRY_JITTER", 0.4)))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(args.input_glob))
    if not files:
        print("[ERROR] No files matched:", args.input_glob)
        sys.exit(1)

    output_format = (
        speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
        if args.use_48k else DEFAULT_OUTPUT_FORMAT
    )

    total_parts = 0
    merged_outputs = []
    for fp in files:
        p = pathlib.Path(fp)
        outs = process_file(
            txt_path=p, out_dir=out_dir,
            voice_host=args.voice_host, voice_sci=args.voice_sci,
            rate=args.rate, max_sents=args.max_sents, max_chars=args.max_chars,
            break_ms=args.break_ms, output_format=output_format,
            strict_clean=args.strict_clean, silence_on_fail=args.silence_on_fail,
            retries=args.retries, base_delay=args.retry_base, jitter=args.retry_jitter
        )
        total_parts += len(outs)
        if args.merge and len(outs) > 1:
            # 注意：新的 part 命名是 _part{i}_{j}，按字典序合并即可
            outs_sorted = sorted(outs, key=lambda pth: pth.name)
            merged = outs_sorted[0].with_name(re.sub(r"_part\d+_\d+$", "", outs_sorted[0].stem) + "_full.mp3")
            if merge_parts_with_ffmpeg(outs_sorted, merged):
                merged_outputs.append(merged)

    if merged_outputs:
        print("\nMerged outputs:")
        for m in merged_outputs:
            print(" -", m)

    if args.only_full_to_docs and merged_outputs:
        docs_dir = pathlib.Path("docs/audio")
        docs_dir.mkdir(parents=True, exist_ok=True)
        for f in merged_outputs:
            target = docs_dir / f.name
            target.write_bytes(f.read_bytes())
            print("[OK] copied", f, "->", target)

    if total_parts == 0:
        print("[ERROR] No MP3 generated.")
        sys.exit(1)

if __name__ == "__main__":
    main()
