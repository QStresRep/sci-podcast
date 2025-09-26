# -*- coding: utf-8 -*-
"""
Azure Speech TTS 批处理脚本（长文分段版，HD 兼容 + 429 限速稳健版）

- 输入：posts/*.txt（第1行=标题；第2行可选 "Date:"；第3行起为正文）
- 支持行首 "Host:" / "Scientist:" 切换声线
- 文本清理、按句分块、逐块合成 MP3；可用 ffmpeg / pydub 合并
- --only-full-to-docs：只把 *_full.mp3 复制到 docs/audio
- ✅ HD 名称（含 ":DragonHD"）自动关闭 <prosody>/<break>，避免 InvalidSsml
- ✅ 429 TooManyRequests：指数退避 + 抖动 + 全局节流（环境变量可调）
- ✅ 单段也会产 *_full.mp3；失败/无产物退出码=1；打印取消详情便于排错
"""

import os, re, html, datetime, pathlib, sys, glob, unicodedata, argparse, subprocess, time, shutil, random
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
VOICE_SCI_DEFAULT  = "en-US-Andrew:DragonHDLatestNeural"
RATE_DEFAULT       = "+20%"  # '20%' 会自动转 '+20%'

ROLE_LINE_PAT = re.compile(r'^(host|scientist)\s*:\s*(.+)$', re.I)
SENT_SPLIT = re.compile(r'(?<=[\.\?\!。！？])\s+')

# ------------------------ 工具函数 ------------------------
def canonicalize_voice(v: str | None, fallback: str) -> str:
    """保留冒号（HD 声线），仅 strip。"""
    if not v:
        return fallback
    v = v.strip()
    return v or fallback

def canonicalize_rate(r: str | None) -> str:
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
            chunks.append(cur); cur = []
            sent_count = 0; char_count = 0
    for voice, sents in items:
        for s in sents:
            s_len = len(s)
            if (sent_count + 1 > max_sents) or (char_count + s_len > max_chars):
                flush()
            if cur and cur[-1][0] == voice:
                cur[-1][1].append(s)
            else:
                cur.append((voice, [s]))
            sent_count += 1; char_count += s_len
    flush()
    return chunks

def is_hd_voice(voice_name: str) -> bool:
    return ":DragonHD" in voice_name

def build_ssml_from_chunk(chunk, rate: str, break_ms: int) -> str:
    parts = []
    for voice, sents in chunk:
        if is_hd_voice(voice):
            # HD：禁用 prosody/break
            inner = "".join("<s>" + html.escape(seg) + "</s>" for seg in sents)
            parts.append(f'<voice name="{html.escape(voice)}">{inner}</voice>')
        else:
            inner = "".join(
                "<s>" + html.escape(seg) + f"</s><break time='{int(break_ms)}ms'/>"
                for seg in sents
            )
            parts.append(
                f'<voice name="{html.escape(voice)}"><prosody rate="{html.escape(rate)}">{inner}</prosody></voice>'
            )
    return '<speak version="1.0" xml:lang="en-US">' + "".join(parts) + "</speak>"

# ------------------------ 合并函数（ffmpeg & pydub 兜底） ------------------------
def merge_parts_with_ffmpeg(parts: List[pathlib.Path], merged_path: pathlib.Path) -> bool:
    """用 ffmpeg concat 合并 MP3；失败返回 False。"""
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
        print("[OK] merged (ffmpeg) ->", merged_path)
        return True
    except Exception as e:
        print("[WARN] ffmpeg merge failed:", e)
        return False

def merge_parts_with_pydub(parts: List[pathlib.Path], merged_path: pathlib.Path) -> bool:
    """无 ffmpeg 时用 pydub 兜底合并。"""
    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for p in parts:
            combined += AudioSegment.from_file(p)
        combined.export(str(merged_path), format="mp3", bitrate="160k")
        print("[OK] merged (pydub) ->", merged_path)
        return True
    except Exception as e:
        print("[WARN] pydub merge failed:", e)
        return False

# ------------------------ 合成 ------------------------
def synth_ssml(ssml: str, out_path: str, prefer_voice_for_config: str, output_format):
    key = os.getenv("SPEECH_KEY"); region = os.getenv("SPEECH_REGION")
    if not key or not region:
        raise SystemExit("Missing SPEECH_KEY / SPEECH_REGION secrets.")
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = prefer_voice_for_config
    speech_config.set_speech_synthesis_output_format(output_format)

    audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    max_retries = int(os.getenv("RETRIES", "10"))
    for attempt in range(1, max_retries + 1):
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return
        # 失败分支
        wait = 0.0
        if result.reason == speechsdk.ResultReason.Canceled:
            cd = result.cancellation_details
            code = getattr(cd, 'error_code', None)
            print(f"[WARN] attempt {attempt} canceled. reason={getattr(cd,'reason',None)} error_code={code}")
            print(f"[WARN] details: {getattr(cd,'error_details','')}")
            if str(code) == "CancellationErrorCode.TooManyRequests":
                wait = min((2 ** attempt) * 3 + random.uniform(0, 2.0), 60.0)
                print(f"[THROTTLE] 429 backoff {wait:.1f}s before retry")
        if wait == 0.0:
            wait = min(1.5 * attempt, 15.0)
        if attempt < max_retries:
            time.sleep(wait)
    raise RuntimeError(f"TTS failed after {max_retries} retries -> {out_path}")

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser(description="Azure Speech TTS batch (long-text segmented).")
    ap.add_argument("--input-glob", default=os.getenv("INPUT_GLOB", DEFAULT_INPUT_GLOB))
    ap.add_argument("--out-dir",    default=os.getenv("OUT_DIR", DEFAULT_OUT_DIR))
    ap.add_argument("--max-sents",  type=int, default=int(os.getenv("MAX_SENTS", DEFAULT_MAX_SENTS)))
    ap.add_argument("--max-chars",  type=int, default=int(os.getenv("MAX_CHARS", DEFAULT_MAX_CHARS)))
    ap.add_argument("--break-ms",   type=int, default=int(os.getenv("BREAK_MS", DEFAULT_BREAK_MS)))
    ap.add_argument("--merge",      action="store_true")
    ap.add_argument("--voice-host", default=canonicalize_voice(os.getenv("VOICE_HOST"), VOICE_HOST_DEFAULT))
    ap.add_argument("--voice-sci",  default=canonicalize_voice(os.getenv("VOICE_SCI"),  VOICE_SCI_DEFAULT))
    ap.add_argument("--rate",       default=os.getenv("SPEED", RATE_DEFAULT))
    ap.add_argument("--use-48k",    action="store_true")
    ap.add_argument("--only-full-to-docs", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    args.rate = canonicalize_rate(args.rate)

    files = sorted(glob.glob(args.input_glob))
    if not files:
        print("[ERROR] No files matched:", args.input_glob); sys.exit(1)

    output_format = (
        speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
        if args.use_48k else DEFAULT_OUTPUT_FORMAT
    )

    # 全局节流参数（可通过环境变量调）
    throttle_ms = int(os.getenv("THROTTLE_MS", "3000"))          # 每块间隔（毫秒）
    pause_every = int(os.getenv("CHUNK_PAUSE_EVERY", "5"))       # 每 N 块长休眠
    pause_secs  = float(os.getenv("CHUNK_PAUSE_SECS", "15"))     # 长休眠秒数

    merged_outputs, failures = [], []

    for fp in files:
        p = pathlib.Path(fp)
        try:
            raw = p.read_text(encoding="utf-8").splitlines()
            if len(raw) < 3:
                print("[WARN] too short:", p); continue
            title = (raw[0] or "Episode").strip()
            date = raw[1].split(":", 1)[1].strip() if raw[1].lower().startswith("date:") else datetime.date.today().isoformat()
            body = sanitize_text("\n".join(raw[2:]).strip())
            if len(body) < 20:
                print("[WARN] body short:", p); continue

            safe_date = slugify(date)
            base_name = safe_date + "_" + slugify(title) + ".mp3"
            base_out  = pathlib.Path(args.out_dir) / base_name

            items  = build_dialog_items(body, args.voice_host, args.voice_sci)
            chunks = chunk_dialog_items(items, max_sents=args.max_sents, max_chars=args.max_chars)
            print("[INFO] chunks=" + str(len(chunks)))

            outs = []
            for idx, chunk in enumerate(chunks, 1):
                ssml = build_ssml_from_chunk(chunk, args.rate, args.break_ms)
                out_path = base_out if len(chunks) == 1 else base_out.with_name(base_out.stem + "_part" + str(idx) + base_out.suffix)
                print("[TTS]", p, "->", out_path)
                synth_ssml(ssml, str(out_path), args.voice_host, output_format)
                outs.append(out_path)

                # 全局节流：每块之间等待；每 N 块再长休眠
                if idx < len(chunks):
                    if throttle_ms > 0:
                        time.sleep(throttle_ms / 1000.0)
                    if pause_every > 0 and (idx % pause_every == 0):
                        print(f"[THROTTLE] periodic sleep {pause_secs:.1f}s after chunk {idx}")
                        time.sleep(pause_secs)

            # 合并/产出 full
            if args.merge and outs:
                if len(outs) > 1:
                    merged = outs[0].with_name(outs[0].stem.replace("_part1", "") + "_full.mp3")
                    ok = merge_parts_with_ffmpeg(outs, merged) or merge_parts_with_pydub(outs, merged)
                    if not ok:
                        shutil.copy2(outs[0], merged)
                        print("[OK] fallback copied first part as full ->", merged)
                    merged_outputs.append(merged)
                else:
                    src = outs[0]; merged = src.with_name(src.stem + "_full.mp3")
                    shutil.copy2(src, merged)
                    print("[OK] single-part copied as full ->", merged)
                    merged_outputs.append(merged)

        except SystemExit as e:
            print("[FAIL]", p, ":", e); failures.append((str(p), str(e)))
        except Exception as e:
            print("[FAIL]", p, ":", e); failures.append((str(p), str(e)))

    # 复制到 docs/audio
    if args.only_full_to_docs and merged_outputs:
        docs_dir = pathlib.Path("docs/audio"); docs_dir.mkdir(parents=True, exist_ok=True)
        for f in merged_outputs:
            try:
                target = docs_dir / f.name
                shutil.copy2(f, target)
                print("[OK] copied", f, "->", target)
            except Exception as e:
                print("[FAIL] copy to docs/audio failed:", e)
                failures.append((str(f), f"copy failed: {e}"))

    if failures:
        print("\nSome files failed:")
        for f, e in failures: print(" -", f, ":", e)
        sys.exit(1)
    if not merged_outputs:
        print("[ERROR] No MP3 generated."); sys.exit(1)

if __name__ == "__main__":
    main()
