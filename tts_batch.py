name: TTS from posts

on:
  workflow_dispatch:
  push:
    paths:
      - "posts/**/*.txt"
      - ".github/workflows/tts-from-posts.yml"
      - "tts_batch.py"

permissions:
  contents: write

jobs:
  tts:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install azure-cognitiveservices-speech
          sudo apt-get update
          sudo apt-get install -y ffmpeg

      # ✅ 使用新参数，只把 *_full.mp3 放到 docs/audio
      - name: Run TTS batch (only copy *_full.mp3 to docs/audio)
        env:
          SPEECH_KEY: ${{ secrets.SPEECH_KEY }}
          SPEECH_REGION: ${{ secrets.SPEECH_REGION }}
          VOICE_HOST: en-US-EmmaMultilingualNeural
          VOICE_SCI: en-US-AndrewMultilingualNeural
          SPEED: "30%"
        run: |
          python tts_batch.py --merge --only-full-to-docs

      # ✅ 强制推送，避免远端更新导致失败
      - name: Commit outputs
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git fetch origin main
          git add docs/audio
          git commit -m "Add merged TTS outputs" || echo "Nothing to commit"
          git pull --rebase origin main || true
          git push origin HEAD:main --force
