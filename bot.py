import asyncio
from enum import Enum
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(room_url: str, token: str):
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import EndFrame, LLMMessagesFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.services.cartesia import CartesiaTTSService
    from pipecat.services.xtts import XTTSService
    from pipecat.services.openai import OpenAILLMService
    from pipecat.transports.services.daily import DailyParams, DailyTransport
    from pipecat.transcriptions.language import Language
    from loguru import logger

    # from openai.types.audio import Transcription
    # from typing import Optional
    # from pipecat.services.ai_services import SegmentedSTTService
    # from pipecat.frames.frames import Frame, TranscriptionFrame, ErrorFrame
    # from pipecat.utils.time import time_now_iso8601
    # from openai import OpenAI
    # from typing import AsyncGenerator

    # class OpenAISTTService(SegmentedSTTService):
    #     def __init__(
    #         self,
    #         *,
    #         model: str = "whisper-1",
    #         api_key: Optional[str] = None,
    #         base_url: Optional[str] = None,
    #         **kwargs,
    #     ):
    #         super().__init__(**kwargs)
    #         self.set_model_name(model)
    #         self._client = OpenAI()

    #     async def set_model(self, model: str):
    #         self.set_model_name(model)

    #     def can_generate_metrics(self) -> bool:
    #         return True

    #     async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
    #         try:
    #             await self.start_processing_metrics()
    #             await self.start_ttfb_metrics()

    #             response: Transcription = (
    #                 await self._client.audio.transcriptions.create(
    #                     file=("audio.wav", audio, "audio/wav"), model=self.model_name
    #                 )
    #             )

    #             await self.stop_ttfb_metrics()
    #             await self.stop_processing_metrics()

    #             text = response.text.strip()

    #             if text:
    #                 logger.debug(f"Transcription: [{text}]")
    #                 yield TranscriptionFrame(text, "", time_now_iso8601())
    #             else:
    #                 logger.warning("Received empty transcription from API")

    #         except Exception as e:
    #             logger.exception(f"Exception during transcription: {e}")
    #             yield ErrorFrame(f"Error during transcription: {str(e)}")
    from pipecat.frames.frames import (
        CancelFrame,
        ErrorFrame,
        Frame,
        StartFrame,
        TranscriptionFrame,
    )
    from pipecat.services.ai_services import SegmentedSTTService
    from pipecat.utils.time import time_now_iso8601
    from typing import AsyncGenerator
    import io
    import wave
    import numpy as np
    class Model(Enum):
        """Available OpenAI Whisper API models"""

        WHISPER_1 = "whisper-1"

    class WhisperAPIService(SegmentedSTTService):
        """Service for OpenAI's Whisper API transcription"""

        def __init__(
            self,
            *,
            api_key: str,
            model: str | Model = Model.WHISPER_1,
            base_url: str = "https://api.openai.com/v1",
            language: Language = None,
            temperature: float = 0,
            prompt: str = None,
            response_format: str = "json",
            sample_rate: int = 16000,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self._api_key = api_key
            self._base_url = base_url.rstrip("/")
            self._session: aiohttp.ClientSession | None = None
            self._sample_rate = sample_rate
            self.set_model_name(model if isinstance(model, str) else model.value)

            self._settings = {
                "model": self.model_name,
                "temperature": temperature,
                "response_format": response_format,
            }

            if language:
                self._settings["language"] = self.language_to_service_language(language)
            if prompt:
                self._settings["prompt"] = prompt

        def can_generate_metrics(self) -> bool:
            return True

        def _prepare_audio(self, audio: bytes) -> bytes:
            """Convert raw PCM audio to WAV format"""
            # Convert to float32 array first (matching whisper.py)
            audio_float = (
                np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Convert back to int16 for WAV
            audio_int16 = (audio_float * 32768).astype(np.int16)

            # Create WAV
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, "wb") as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self._sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                return wav_io.getvalue()

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            """Transcribes given audio using Whisper API"""
            try:
                await self.start_processing_metrics()
                await self.start_ttfb_metrics()

                # Prepare audio in the same way as local Whisper
                wav_audio = self._prepare_audio(audio)

                # Create session for this request
                async with aiohttp.ClientSession(
                    headers={"Authorization": f"Bearer {self._api_key}"}
                ) as session:
                    # Prepare form data
                    form = aiohttp.FormData()
                    form.add_field(
                        "file",
                        wav_audio,
                        filename="audio.wav",
                        content_type="audio/wav",
                    )

                    # Add settings as form fields
                    for key, value in self._settings.items():
                        if value is not None:
                            form.add_field(key, str(value))

                    # Make API request
                    async with session.post(
                        f"{self._base_url}/audio/transcriptions", data=form
                    ) as response:
                        await self.stop_ttfb_metrics()

                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Whisper API error: {response.status} - {error_text}"
                            )
                            yield ErrorFrame(f"Transcription failed: {error_text}")
                            return

                        result = await response.json()
                        text = result.get("text", "").strip()

                        if text:
                            logger.debug(f"Transcription: [{text}]")
                            yield TranscriptionFrame(text, "", time_now_iso8601())

            except Exception as e:
                logger.exception(f"Whisper API error: {e}")
                yield ErrorFrame(f"Transcription failed: {str(e)}")
            finally:
                await self.stop_processing_metrics()

        def language_to_service_language(self, language: Language) -> str:
            """Convert internal language enum to ISO-639-1 codes"""
            return str(language.value).split("-")[0].lower()
    class SealionLLMService(OpenAILLMService):
        """A service for interacting with Groq's API using the OpenAI-compatible interface.

        This service extends OpenAILLMService to connect to Groq's API endpoint while
        maintaining full compatibility with OpenAI's interface and functionality.

        Args:
            api_key (str): The API key for accessing Groq's API
            base_url (str, optional): The base URL for Groq API. Defaults to "https://api.groq.com/openai/v1"
            model (str, optional): The model identifier to use. Defaults to "llama-3.1-70b-versatile"
            **kwargs: Additional keyword arguments passed to OpenAILLMService
        """

        def __init__(
            self,
            *,
            api_key: str,
            base_url: str = "https://api.sea-lion.ai/v1",
            model: str = "aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct",
            **kwargs,
        ):
            super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

        def create_client(self, api_key=None, base_url=None, **kwargs):
            """Create OpenAI-compatible client for SEALION API endpoint."""
            logger.debug(f"Creating SEALION client with api {base_url}")
            return super().create_client(api_key, base_url, **kwargs)

    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "bot",
            DailyParams(
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        # stt = WhisperAPIService(api_key=os.getenv("OPENAI_API_KEY"), model="whisper-1")
        stt = WhisperAPIService(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model="whisper-large-v3-turbo",
        )

        # tts = CartesiaTTSService(
        #     api_key=os.getenv("CARTESIA_API_KEY", ""), voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22"
        # )
        tts = XTTSService(
            aiohttp_session=session,
            voice_id="Ana Florence",  # Marcos Rudaski
            language=Language.EN,
            # base_url="http://13.59.71.92:8000",  # A10G us-east-2
            base_url="http://35.94.29.191:8000",  # L40s us-west-2
        )

        # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        llm = SealionLLMService(
            api_key=os.getenv("AISG_API_KEY"),
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)


def _voice_bot_process(room_url: str, token: str):
    asyncio.run(main(room_url, token))
