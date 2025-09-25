# -*- coding: utf-8 -*-
"""
Azure Speech TTS 批处理脚本（长文分段版，带超限兜底）
- 默认分块更小 (max_sents=40, max_chars=3000)
- 自动检测 SSML 长度超过 AZURE_SSML_LIMIT 时再细分
- 确保不会因为超长而触发 Azure Canceled
- 失败时重试；可选合并；支持只把 *_full.mp3 放到 docs/audio
"""

import os, re, html, datetime, pathlib, sys, glob, unicodedata, argparse, subprocess, time, random
from typing import List, Tuple

try:
    import azure.cognitiveservices.speech as speechsdk
except Exception as e:
    print("[ERROR] Missing package 'azure-cognitiveservices-speech':", e)
    sys.exit(1)

# =========================
# 默认参数
# =========================
DEFAULT_INPUT_GLOB = "posts/*.txt"
DEFAULT_OUT_DIR    = "tts_out"
DEFAULT_MAX_SENTS  = 40     # ⚠️ 比原来小
DEFAULT_MAX_CHARS  = 3000   # ⚠️ 比原来小
DEFAULT_BREAK_MS   = 250

DEFAULT_OUTPUT_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3

VOICE_HOST_DEFAULT = "en-US-EmmaMultilingualNeural"
VOICE_SCI_DEFAULT  = "en-US-AndrewMultilingualNeural"
RATE_DEFAULT       = "30%"

# ✅ Azure 官方限制 ~5000 字符，这里留 buffer
AZURE_SSML_LIMIT = 4500

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
    for z in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"):
        s = s.replace(z, "")
    s = "".join(ch for ch in s if (ch >= " " or ch in "\n\t"))
    return unicodedata.normalize("NFC", s)

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
    chunks = []
    cur = []
    sent_count = 0
    char_count = 0

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

def split_chunk_by_chars(chunk_pairs, rate, break_ms, limit):
    """把超长的 chunk 再按字符数切分"""
    small_chunks, cur, cur_len = [], [], 0
    def sent_to_len(s): return len(s) + 32
    for voice, sents in chunk_pairs:
        for s in sents:
            add_len = sent_to_len(s)
            if cur and cur_len + add_len > limit:
                small_chunks.append(cur)
                cur, cur_len = [], 0
            if cur and cur[-1][0] == voice:
                cur[-1][1].append(s)
            else:
                cur.append((voice, [s]))
            cur_len += add_len
    if cur:
        small_chunks.append(cur)
    return small_chunks

def synth_ssml(ssml: str, out_path: str, prefer_voice_for_config: str, output_format,
               retries=5, retry_base=1.5, retry_jitter=0.5):
    key = os.getenv("SPEECH_KEY"); region = os.getenv("SPEECH_REGION")
    if not key or not region:
        raise SystemExit("Missing SPEECH_KEY / SPEECH_REGION secrets.")
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = prefer_voice_for_config
    speech_config.set_speech_synthesis_output_format(output_format)

    audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    for attempt in range(1, retries + 1):
        print(f"[DEBUG] synth attempt={attempt} ssml_len={len(ssml)} -> {out_path}")
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return
        print(f"[WARN] attempt {attempt} failed, reason={result.reason}")
        if attempt < retries:
            delay = retry_base * (2 ** (attempt - 1))
            delay += random.uniform(-retry_jitter, retry_jitter) * delay
            delay = max(1.0, delay)
            print(f"[INFO] retrying after {delay:.2f}s ...")
            time.sleep(delay)
        else:
            raise RuntimeError(f"TTS failed after {retries} retries -> {out_path}")

def process_file(txt_path: pathlib.Path, out_dir: pathlib.Path, voice_host: str, voice_sci: str,
                 rate: str, max_sents: int, max_chars: int, break_ms: int, output_format,
                 retries=5, retry_base=1.5, retry_jitter=0.5):
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

    items  = build_dialog_items(body, voice_host, voice_sci)
    chunks = chunk_dialog_items(items, max_sents=max_sents, max_chars=max_chars)
    outputs = []

    print("[INFO] chunks=" + str(len(chunks)))

    for idx, chunk in enumerate(chunks, 1):
        ssml = build_ssml_from_chunk(chunk, rate, break_ms)
        # ⚠️ 如果超长，再切
        if len(ssml) > AZURE_SSML_LIMIT:
            sub_chunks = split_chunk_by_chars(chunk, rate, break_ms, AZURE_SSML_LIMIT - 500)
            for j, sub in enumerate(sub_chunks, 1):
                sub_ssml = build_ssml_from_chunk(sub, rate, break_ms)
                out_path = base_out.with_name(f"{base_out.stem}_part{idx}_{j}{base_out.suffix}")
                print("[TTS]", txt_path, "->", out_path)
                synth_ssml(sub_ssml, str(out_path), voice_host, output_format, retries, retry_base, retry_jitter)
                print("[OK] wrote", out_path, "(", out_path.stat().st_size, "bytes )")
                outputs.append(out_path)
            continue

        out_path = base_out if len(chunks) == 1 else base_out.with_name(base_out.stem + "_part" + str(idx) + base_out.suffix)
        print("[TTS]", txt_path, "->", out_path)
        synth_ssml(ssml, str(out_path), voice_host, output_format, retries, retry_base, retry_jitter)
        print("[OK] wrote", out_path, "(", out_path.stat().st_size, "bytes )")
        outputs.append(out_path)

    return outputs

def merge_parts_with_ffmpeg(parts: List[pathlib.Path], merged_path: pathlib.Path):
    if not parts:
        return False
    try:
        concat_arg = "concat:" + "|".join(str(p) for p in parts)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", concat_arg, "-c", "copy", str(merged_path)],
            check=True
        )
        print("[OK] merged ->", merged_path)
        return True
    except Exception as e:
        print("[WARN] fast concat failed, try re-encode:", e)

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
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--retry-base", type=float, default=1.5)
    ap.add_argument("--retry-jitter", type=float, default=0.5)
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
    failures = []

    for fp in files:
        p = pathlib.Path(fp)
        try:
            outs = process_file(
                txt_path=p, out_dir=out_dir,
                voice_host=args.voice_host, voice_sci=args.voice_sci,
                rate=args.rate, max_sents=args.max_sents, max_chars=args.max_chars,
                break_ms=args.break_ms, output_format=output_format,
                retries=args.retries, retry_base=args.retry_base, retry_jitter=args.retry_jitter
            )
            total_parts += len(outs)
            if args.merge and len(outs) > 1:
                merged = outs[0].with_name(outs[0].stem.replace("_part1", "") + "_full.mp3")
                if merge_parts_with_ffmpeg(outs, merged):
                    merged_outputs.append(merged)
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
            target.write_bytes(f.read_bytes())
            print("[OK] copied", f, "->", target)

    if total_parts == 0:
        print("[ERROR] No MP3 generated.")
        sys.exit(1)

if __name__ == "__main__":
    main()
