"""TTS Model Manager for handling different TTS models."""

import asyncio
import logging
import subprocess
from typing import Dict, Optional
import requests
import os
import signal

logger = logging.getLogger(__name__)

class TTSServer:
	"""Represents a running TTS server instance."""
	
	def __init__(self, model_name: str, port: int):
		self.model_name = model_name
		self.port = port
		self.process: Optional[subprocess.Popen] = None
		self.url = f"http://localhost:{port}"

	async def start(self, venv_path: str) -> bool:
		"""Start the TTS server."""
		try:
			# Kill any existing process on the same port
			await self._kill_process_on_port(self.port)
			
			# Construct and execute the command
			activate_cmd = f"source {os.path.join(venv_path, 'bin/activate')}"
			server_cmd = f"tts-server --model_name '{self.model_name}' --use_cuda --port {self.port}"
			full_cmd = f"{activate_cmd} && {server_cmd}"
			
			# Start the server process
			self.process = subprocess.Popen(
				full_cmd,
				shell=True,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				preexec_fn=os.setsid  # Create a new process group
			)
			
			# Wait for server to start
			for _ in range(30):  # Wait up to 30 seconds
				try:
					response = requests.get(f"{self.url}/health")
					if response.status_code == 200:
						logger.info(f"TTS server for {self.model_name} started on port {self.port}")
						return True
				except requests.exceptions.ConnectionError:
					await asyncio.sleep(1)
					
				# Check if process has failed
				if self.process.poll() is not None:
					stderr = self.process.stderr.read()
					logger.error(f"TTS server failed to start: {stderr}")
					return False
			
			logger.error(f"Timeout waiting for TTS server {self.model_name} to start")
			await self.stop()  # Clean up the process
			return False
			
		except Exception as e:
			logger.error(f"Failed to start TTS server {self.model_name}: {str(e)}")
			return False

	async def stop(self) -> bool:
		"""Stop the TTS server."""
		try:
			if self.process:
				# Kill the entire process group
				os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
				try:
					self.process.wait(timeout=5)
				except subprocess.TimeoutExpired:
					os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
				self.process = None
			return True
		except Exception as e:
			logger.error(f"Failed to stop TTS server {self.model_name}: {str(e)}")
			return False

	async def _kill_process_on_port(self, port: int) -> None:
		"""Kill any existing process running on the specified port."""
		try:
			# Find process using the port
			result = subprocess.run(
				f"lsof -i :{port} -t",
				shell=True,
				capture_output=True,
				text=True
			)
			
			if result.stdout:
				pid = int(result.stdout.strip())
				try:
					os.kill(pid, signal.SIGTERM)
					await asyncio.sleep(1)  # Give it a second to terminate
					try:
						os.kill(pid, 0)  # Check if process still exists
						os.kill(pid, signal.SIGKILL)  # Force kill if still running
					except OSError:
						pass  # Process already terminated
				except ProcessLookupError:
					pass  # Process already gone
		except Exception as e:
			logger.warning(f"Failed to kill process on port {port}: {str(e)}")

	async def synthesize(self, text: str, speaker_name: Optional[str] = None) -> Optional[bytes]:
		"""Synthesize text using the TTS server."""
		try:
			params = {"text": text}
			if speaker_name:
				params["speaker_name"] = speaker_name
			
			response = requests.get(f"{self.url}/api/tts", params=params)
			if response.status_code == 200:
				return response.content
			else:
				logger.error(f"Synthesis failed: {response.text}")
				return None
		except Exception as e:
			logger.error(f"Failed to synthesize text: {str(e)}")
			return None

class TTSModelManager:
	"""Manages the loading and unloading of TTS models."""
	
	def __init__(self, venv_path: str = "~/test/tts/.venv"):
		self.venv_path = os.path.expanduser(venv_path)
		self.base_port = 5002
		self.models: Dict[str, dict] = {
			"xtts_v2": {
				"name": "XTTS v2",
				"model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
				"loaded": False,
				"instance": None
			},
			"bark": {
				"name": "Bark",
				"model_name": "tts_models/multilingual/multi-dataset/bark",
				"loaded": False,
				"instance": None
			}
		}
		self.active_model: Optional[str] = None

	async def load_model(self, model_id: str) -> bool:
		"""Load a specific TTS model."""
		if model_id not in self.models:
			raise ValueError(f"Unknown model: {model_id}")

		if self.active_model:
			# Unload currently active model
			await self.unload_model(self.active_model)

		try:
			model = self.models[model_id]
			port = self.base_port + len([m for m in self.models.values() if m["loaded"]])
			
			server = TTSServer(model["model_name"], port)
			if await server.start(self.venv_path):
				model["loaded"] = True
				model["instance"] = server
				self.active_model = model_id
				logger.info(f"Loaded model: {model_id}")
				return True
			return False
		except Exception as e:
			logger.error(f"Failed to load model {model_id}: {str(e)}")
			return False

	async def unload_model(self, model_id: str) -> bool:
		"""Unload a specific TTS model."""
		if model_id not in self.models:
			raise ValueError(f"Unknown model: {model_id}")

		if not self.models[model_id]["loaded"]:
			return True

		try:
			model = self.models[model_id]
			if model["instance"]:
				await model["instance"].stop()
			
			model["loaded"] = False
			model["instance"] = None
			
			if self.active_model == model_id:
				self.active_model = None
			
			logger.info(f"Unloaded model: {model_id}")
			return True
		except Exception as e:
			logger.error(f"Failed to unload model {model_id}: {str(e)}")
			return False

	async def synthesize(self, text: str, model_id: Optional[str] = None, speaker_name: Optional[str] = None) -> Optional[bytes]:
		"""Synthesize text using the specified or active model."""
		model_id = model_id or self.active_model
		if not model_id:
			raise ValueError("No active model")
		
		if model_id not in self.models:
			raise ValueError(f"Unknown model: {model_id}")
		
		model = self.models[model_id]
		if not model["loaded"] or not model["instance"]:
			raise ValueError(f"Model {model_id} is not loaded")
		
		return await model["instance"].synthesize(text, speaker_name)

	def get_active_model(self) -> Optional[str]:
		"""Get the currently active model ID."""
		return self.active_model

	def list_models(self) -> Dict[str, dict]:
		"""List all available models and their status."""
		return {
			model_id: {
				"name": info["name"],
				"loaded": info["loaded"],
				"port": info["instance"].port if info["instance"] else None
			}
			for model_id, info in self.models.items()
		}
