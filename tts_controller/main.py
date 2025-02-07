from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import uvicorn
import logging

logger = logging.getLogger(__name__)

from .models import TTSModelManager

app = FastAPI(title="TTS Controller")
model_manager = TTSModelManager()

# Mount static files
app.mount("/static", StaticFiles(directory="tts_controller/static"), name="static")
templates = Jinja2Templates(directory="tts_controller/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
	"""Render the home page with TTS control panel."""
	return templates.TemplateResponse(
		"index.html",
		{"request": request, "title": "TTS Controller", "models": model_manager.list_models()}
	)

@app.get("/list_models")
async def list_models():
	"""List all available TTS models."""
	return {"models": model_manager.list_models()}

@app.post("/load_model/{model_id}")
async def load_model(model_id: str):
	"""Load a specific TTS model."""
	try:
		success = await model_manager.load_model(model_id)
		if success:
			return {"status": "success", "message": f"Model {model_id} loaded"}
		else:
			raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}")
	except ValueError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload_model/{model_id}")
async def unload_model(model_id: str):
	"""Unload a specific TTS model."""
	try:
		success = await model_manager.unload_model(model_id)
		if success:
			return {"status": "success", "message": f"Model {model_id} unloaded"}
		else:
			raise HTTPException(status_code=500, detail=f"Failed to unload model {model_id}")
	except ValueError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/synthesize")
async def synthesize(
	text: str,
	model_id: Optional[str] = None,
	speaker_id: Optional[str] = None,
	language_id: Optional[str] = None,
	style_wav: Optional[str] = None
) -> Response:
	"""Synthesize text to speech."""
	try:
		audio_data = await model_manager.synthesize(text, model_id, speaker_id, language_id)
		if audio_data:
			return Response(content=audio_data, media_type="audio/wav")
		return Response(status_code=500, content="Failed to synthesize text")
	except Exception as e:
		logger.error(f"Error during synthesis: {str(e)}")
		return Response(status_code=500, content=str(e))

if __name__ == "__main__":
	uvicorn.run("tts_controller.main:app", host="0.0.0.0", port=8000, reload=True)
