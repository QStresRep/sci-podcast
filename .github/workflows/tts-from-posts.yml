name: TTS from posts

on:
  workflow_dispatch:

# 同一分支互斥，避免并发触发额外 429
concurrency:
  group: tts-from-posts-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install azure-cognitiveservices-speech pydub
          sudo apt-get update
          sudo apt-get install -y ffmpeg jq

      - name: Verify Azure secrets
        run: |
          echo "SPEECH_REGION=${{ secrets.SPEECH_REGION }}"
          test -n "${{ secrets.SPEECH_KEY }}" || (echo "Missing SPEECH_KEY" && exit 1)
          test -n "${{ secrets.SPEECH_REGION }}" || (echo "Missing SPEECH_REGION" && exit 1)

      # 🔎 冒烟测试（REST，一句“Hello”）
      - name: Smoke test TTS (REST, eastus)
        env:
          SPEECH_KEY: ${{ secrets.SPEECH_KEY }}
          SPEECH_REGION: ${{ secrets.SPEECH_REGION }}   # ← 确保是 eastus
        run: |
          set -e
          echo "Using region: ${SPEECH_REGION}"
          printf "%s\n" \
            "<speak version='1.0' xml:lang='en-US'>" \
            "  <voice name='en-US-EmmaMultilingualNeural'>Hello from Emma in eastus.</voice>" \
            "</speak>" > hello.ssml
          curl -sS -f \
            -H "Ocp-Apim-Subscription-Key: ${SPEECH_KEY}" \
            -H "Content-Type: application/ssml+xml" \
            -H "X-Microsoft-OutputFormat: audio-24khz-160kbitrate-mono-mp3" \
            --data-binary @hello.ssml \
            "https://${SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1" \
            -o hello.mp3
          ls -alh hello.mp3

      - name: Debug posts folder
        run: |
          echo "Listing posts/"
          ls -alh posts || echo "No posts folder!"
          echo "Show first file head (if any):"
          for f in posts/*.txt; do
            [ -e "$f" ] || continue
            echo "---- $f ----"
            head -n 5 "$f"
            break
          done

      # 全 Multilingual，配合适度节流（与 tts_batch.py 对应）
      - name: Run TTS batch (multilingual, throttled, eastus)
        env:
          SPEECH_KEY: ${{ secrets.SPEECH_KEY }}
          SPEECH_REGION: ${{ secrets.SPEECH_REGION }}

          VOICE_HOST: en-US-Emma:DragonHDLatestNeural
          VOICE_SCI:  en-US-Andrew:DragonHDLatestNeural
          SPEED: +20%

          # 节流参数（稳）
          THROTTLE_MS: "6000"       # 每块间隔 6s
          CHUNK_PAUSE_EVERY: "3"    # 每 3 块长休眠
          CHUNK_PAUSE_SECS: "30"    # 长休眠 30s
          RETRIES: "12"             # 每块最多重试 12 次
        run: |
          # 块稍大以减少请求次数
          python tts_batch.py --merge --max-chars 3600 --max-sents 120 --break-ms 300 --only-full-to-docs
          echo "List outputs:"
          ls -alh tts_out || true
          ls -alh docs/audio || true

      - name: Commit merged audio
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add docs/audio
          git commit -m "Add merged TTS outputs" || echo "Nothing to commit"
          git push
