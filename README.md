# Deploying Pipecat to Modal.com

Barebones deployment example for [modal.com](https://www.modal.com)

1. Install dependencies

```bash
python -m venv venv
source venv/bin/active # or OS equivalent
pip install -r requirements.txt
```

2. Setup .env

```bash
cp env.example .env
```

Alternatively, you can configure your Modal app to use [secrets](https://modal.com/docs/guide/secrets)

3. Test the app locally

```bash
modal serve app.py # run the server
curl -X POST https://{modal_dev_url} # POST request to create Daily Room
```

4. Deploy to production

```bash
modal deploy app.py
```

## Configuration options

This app sets some sensible defaults for reducing cold starts, such as `minkeep_warm=1`, which will keep at least 1 warm instance ready for your bot function.

It has been configured to only allow a concurrency of 1 (`max_inputs=1`) as each user will require their own running function.

# OpenAI Compatible Whisper Servers
* [Faster Whisper Server](https://github.com/fedirz/faster-whisper-server)
* [More barebones version](https://github.com/matatonic/openedai-whisper)
* [another small version](https://github.com/morioka/tiny-openai-whisper-api)

# Whisper Streaming Servers
* https://github.com/ufal/whisper_streaming?tab=readme-ov-file
* https://github.com/ggoonnzzaallo/llm_experiments/blob/main/streamed_text_plus_streamed_audio.py
* https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/openai_whisper/streaming/main.py

* Other STT
* Modal [Seamless M4T](https://modal.com/docs/examples/seamless-chat), [code](https://github.com/modal-labs/seamless-chat/blob/main/seamless.py)

# Other TTS
* [Parler TTS](https://github.com/huggingface/parler-tts)

References
* [Moshi](https://github.com/modal-labs/quillman/blob/main/src/moshi.py)