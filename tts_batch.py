# -*- coding: utf-8 -*-
"""
Azure Speech TTS 批处理脚本（长文分段版，HD 兼容稳健版）

要点：
- 支持 posts/*.txt：第1行标题；第2行可选 "Date:"；第3行起正文
- 支持 Host:/Scientist: 分角色说话
- 自动清理非法字符、按句分块；逐块合成 MP3；可用 ffmpeg 合并
- --only-full-to-docs：只复制 *_full.mp3 到 docs/audio
- ✅ HD 兼容：若 voice 名含 ":DragonHD"（冒号 HD 声线），则不使用 <prosody>/<break>，避免 InvalidSsml
- ✅ 保留冒号：不再截断含冒号的 voice 名
- ✅ 速率规范：'20%' 自动转 '+20%'；HD 分支忽略速率（不套 <prosody>）
- ✅ 单段也会产 *_full.mp3
- ✅ 打印 CancellationDetails（error_code/details）
- ✅ 无产物或失败则退出码=1
"""

import os, re, html, datetime, pathlib, sys, glob, unicodedata, argparse, subprocess, time, shutil
from typing import List, Tuple

try:
    import azure.cognitiveservices.speech as speechsdk
except Exception as e:
    print("[ERROR] Missing package 'azure-cognitiveservices-speech':", e)
    sys.exit(1)

# ------------------------ 默认参数 ------------------------
DEFAULT_INPUT_GLOB = "posts/*.txt"
DEFAULT_OUT_DIR    = "tts_out"
DEFAULT_MAX_SENTS  = 100
DEFAULT_MAX_CHARS  = 3500
DEFAULT_BREAK_MS   = 250

DEFAULT_OUTPUT_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3

VOICE_HOST_DEFAULT = "en-US-Emma:DragonHDLatestNeural"
VOICE_SCI_DEFAULT  = "en-US-AndrewMultilingualNeural"
RATE_DEFAULT       = "+20%"  # 若传入 "20%" 自动转成 "+20%"

ROLE_LINE_PAT = re.compile(r'^(host|scientist)\s*:\s*(.+)$', re.I)
SENT_SPLIT = re.compile(r'(?<=[\.\?\!。！？])\s+')

# ------------------------ 工具函数 ------------------------
def canonicalize_voice(v: str | None, fallback: str) -> str:
    """保持原样（尤其是冒号 HD 名称），仅去除前后空白；不再截断冒号。"""
    if not v:
        return fallback
    v = v.strip()
    # 旧逻辑会切掉冒号后的部分，这里不要这么做
    return v or fallback

def canonicalize_rate(r: str | None) -> str:
    """把 '20%' 正常化为 '+20%'；允许 '+10%', '-10%', '0%'，其余原样返回。"""
    if not r:
        return RATE_DEFAULT
    r = r.strip()
    if re.fullmatch(r'[+-]?\d+%', r):
        return r if r.startswith(("+", "-")) else ("+" + r)
    return r

def sanitize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    for z in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"):
        s = s.replace(z, "")
    s = "".join(ch for ch in s if (ch >= " " or ch in "\n\t"))
    return unicodedata.normalize("NFC", s)

def slugify(s: str) -> str:
    s = s.strip().lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return (s[:80].strip("-")) or "episode"

def parse_role_line(line: str, voice_host: str, voice_sci: str) -> Tuple[str, str]:
    m = ROLE_LINE_PAT.match(line)
    if m:
        role = m.group(1).lower()
        content = m.group(2).strip()
        voice = voice_host if role == "host" else voice_sci
        return voice, content
    return voice_host, line.strip()

def to_sentences(text: str):
    return [seg.strip() for seg in SENT_SPLIT.split(text) if seg and seg.strip()]

def build_dialog_items(body: str, voice_host: str, voice_sci: str):
    items = []
    for raw_ln in body.split("\n"):
        ln = raw_ln.strip()
        if not ln:
            continue
        voice, content = parse_role_line(ln, voice_host, voice_sci)
        sents = to_sentences(content) or [content]
        items.append((voice, sents))
    return items

def chunk_dialog_items(items, max_sents: int, max_chars: int):
    chunks, cur = [], []
    sent_count = char_count = 0

    def flush():
        nonlocal cur, sent_count, char_count
        if cur:
            chunks.append(cur)
            cur = []
            sent_count = 0
            char_count = 0

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

def is_hd_voice(voice_name: str) -> bool:
    """简单判断：含 ':DragonHD' 的视为 HD 声线。"""
    return ":DragonHD" in voice_name

def build_ssml_from_chunk(chunk, rate: str, break_ms: int) -> str:
    parts = []
    for voice, sents in chunk:
        if is_hd_voice(voice):
            # ⚠️ HD：不要用 <prosody> / <break>，避免 InvalidSsml
            inner = "".join("<s>" + html.escape(seg) + "</s>" for seg in sents)
            parts.append(f'<voice name="{html.escape(voice)}">{inner}</voice>')
        else:
            # 常规：可用 prosody / break
            inner = "".join(
                "<s>" + html.escape(seg) + f"</s><break time='{int(break_ms)}ms'/>"
                for seg in sents
            )
            parts.append(
                f'<voice name="{html.escape(voice)}"><prosody rate="{html.escape(rate)}">{inner}</prosody></voice>'
            )
    return '<speak version="1.0" xml:lang="en-US">' + "".join(parts) + "</speak>"

def synth_ssml(ssml: str, out_path: str, prefer_voice_for_config: str, output_format):
    key = os.getenv("SPEECH_KEY"); region = os.getenv("SPEECH_REGION")
    if not key or not region:
        raise SystemExit("Missing SPEECH_KEY / SPEECH_REGION secrets.")
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    # 这里设置一个默认 voice；具体 SSML 内部会按 <voice name="..."> 覆盖
    speech_config.speech_synthesis_voice_name = prefer_voice_for_config
    speech_config.set_speech_synthesis_output_format(output_format)

    audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return
        else:
            if result.reason == speechsdk.ResultReason.Canceled:
                cd = result.cancellation_details
                print(f"[WARN] attempt {attempt} canceled. reason={getattr(cd,'reason',None)} "
                      f"error_code={getattr(cd,'error_code',None)}")
                print(f"[WARN] details: {getattr(cd,'error_details','')}")
            else:
                print(f"[WARN] attempt {attempt} failed, reason={result.reason}")
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"TTS failed after {max_retries} retries -> {out_path}")

def process_file(txt_path: pathlib.Path, out_dir: pathlib.Path, voice_host: str, voice_sci: str,
                 rate: str, max_sents: int, max_chars: int, break_ms: int, output_format):
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

    safe_date = slugify(date)  # 例如 "September 25, 2025" -> "september-25-2025"
    base_name = safe_date + "_" + slugify(title) + ".mp3"
    base_out = out_dir / base_name

    items  = build_dialog_items(body, voice_host, voice_sci)
    chunks = chunk_dialog_items(items, max_sents=max_sents, max_chars=max_chars)
    outputs = []

    print("[INFO] chunks=" + str(len(chunks)))

    for idx, chunk in enumerate(chunks, 1):
        ssml = build_ssml_from_chunk(chunk, rate, break_ms)
        out_path = base_out if len(chunks) == 1 else base_out.with_name(base_out.stem + "_part" + str(idx) + base_out.suffix)
        print("[TTS]", txt_path, "->", out_path)
        synth_ssml(ssml, str(out_path), voice_host, output_format)
        outputs.append(out_path)
        time.sleep(0.2)
    return outputs

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

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser(description="Azure Speech TTS batch (long-text segmented).")
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
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 规范化 rate（'20%' -> '+20%'）
    args.rate = canonicalize_rate(args.rate)

    files = sorted(glob.glob(args.input_glob))
    if not files:
        print("[ERROR] No files matched:", args.input_glob)
        sys.exit(1)

    output_format = (
        speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
        if args.use_48k else DEFAULT_OUTPUT_FORMAT
    )

    merged_outputs = []
    failures = []

    for fp in files:
        p = pathlib.Path(fp)
        try:
            outs = process_file(
                txt_path=p, out_dir=out_dir,
                voice_host=args.voice_host, voice_sci=args.voice_sci,
                rate=args.rate, max_sents=args.max_sents, max_chars=args.max_chars,
                break_ms=args.break_ms, output_format=output_format
            )
            if args.merge and outs:
                if len(outs) > 1:
                    merged = outs[0].with_name(outs[0].stem.replace("_part1", "") + "_full.mp3")
                    if merge_parts_with_ffmpeg(outs, merged):
                        merged_outputs.append(merged)
                else:
                    src = outs[0]
                    merged = src.with_name(src.stem + "_full.mp3")
                    try:
                        shutil.copy2(src, merged)
                        print("[OK] single-part copied as full ->", merged)
                        merged_outputs.append(merged)
                    except Exception as e:
                        print("[FAIL] single-part copy failed:", e)
                        failures.append((str(p), f"single-part copy failed: {e}"))

        except SystemExit as e:
            print("[FAIL]", p, ":", e)
            failures.append((str(p), str(e)))
        except Exception as e:
            print("[FAIL]", p, ":", e)
            failures.append((str(p), str(e)))

    if merged_outputs:
        print("\nMerged outputs:")
        for m in merged_outputs:
            print(" -", m)

    if args.only_full_to_docs and merged_outputs:
        docs_dir = pathlib.Path("docs/audio")
        docs_dir.mkdir(parents=True, exist_ok=True)
        for f in merged_outputs:
            target = docs_dir / f.name
            try:
                shutil.copy2(f, target)
                print("[OK] copied", f, "->", target)
            except Exception as e:
                print("[FAIL] copy to docs/audio failed:", e)
                failures.append((str(f), f"copy failed: {e}"))

    if failures:
        print("\nSome files failed:")
        for f, e in failures:
            print(" -", f, ":", e)
        sys.exit(1)

    if not merged_outputs:
        print("[ERROR] No MP3 generated.")
        sys.exit(1)

if __name__ == "__main__":
    main()
