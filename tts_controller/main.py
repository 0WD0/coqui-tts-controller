from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import uvicorn
from io import BytesIO

from .models import TTSModelManager

app = FastAPI(title="TTS Controller")
model_manager = TTSModelManager()

# Mount static files
app.mount("/static", StaticFiles(directory="tts_controller/static"), name="static")
templates = Jinja2Templates(directory="tts_controller/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
	"""Render the home page with TTS control panel."""
	return templates.TemplateResponse(
		"index.html",
		{"request": request, "title": "TTS Controller"}
	)

@app.get("/api/models")
async def list_models():
	"""List all available TTS models."""
	return {"models": model_manager.list_models()}

@app.post("/api/models/{model_id}/load")
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

@app.post("/api/models/{model_id}/unload")
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

@app.post("/api/synthesize")
async def synthesize_text(text: str, model_id: Optional[str] = None, speaker_name: Optional[str] = None):
	"""Synthesize text using the specified model."""
	try:
		audio_data = await model_manager.synthesize(text, model_id, speaker_name)
		if audio_data:
			return StreamingResponse(
				BytesIO(audio_data),
				media_type="audio/wav",
				headers={"Content-Disposition": "attachment; filename=synthesized.wav"}
			)
		else:
			raise HTTPException(status_code=500, detail="Failed to synthesize text")
	except ValueError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
	uvicorn.run("tts_controller.main:app", host="0.0.0.0", port=8000, reload=True)
