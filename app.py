import os
import base64
import asyncio.base_events
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
import gradio as gr
from gradio.routes import App as GradioApp

def _safe_del(self):
    try:
        if not self.is_closed():
            self.close()
    except Exception:
        pass

asyncio.base_events.BaseEventLoop.__del__ = _safe_del

try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    gpu_decorator = lambda fn: fn

os.makedirs("static/audio", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

async def _handle_simplify(request: Request):
    from app.main import TextSimplifyRequest, simplify_text_endpoint
    from app.database import SessionLocal
    body = await request.json()
    req_obj = TextSimplifyRequest(**body)
    db = SessionLocal()
    try:
        result = simplify_text_endpoint(req_obj, db)
        audio_url = result.get("audio_narration_url") or ""
        if audio_url.startswith("/static/"):
            local_path = audio_url.lstrip("/")
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    b64_audio = base64.b64encode(f.read()).decode("ascii")
                result["audio_narration_url"] = f"data:audio/mp3;base64,{b64_audio}"
        return JSONResponse(result)
    finally:
        db.close()

def _handle_analytics():
    from app.main import get_analytics_report
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        res = get_analytics_report(db)
        if res.get("chart_url"):
            res["chart_url"] = "/api/analytics/chart.png"
        return JSONResponse(res)
    finally:
        db.close()

def _serve_chart_png():
    chart_path = os.path.join("static", "plots", "readability_evaluation_report.png")
    return FileResponse(chart_path, media_type="image/png")

_orig_create_app = GradioApp.create_app

def _custom_create_app(*args, **kwargs):
    fastapi_app = _orig_create_app(*args, **kwargs)
    fastapi_app.add_api_route("/ui", lambda: FileResponse("static/index.html"), methods=["GET"])
    fastapi_app.add_api_route("/api/simplify/text", _handle_simplify, methods=["POST"])
    fastapi_app.add_api_route("/api/analytics/report", _handle_analytics, methods=["GET"])
    fastapi_app.add_api_route("/api/analytics/chart.png", _serve_chart_png, methods=["GET"])
    return fastapi_app

GradioApp.create_app = staticmethod(_custom_create_app)

@gpu_decorator
def _zero_gpu_ready():
    return "ready"

with gr.Blocks(title="ClearWords | Multilingual Accessibility Engine") as demo:
    gr.HTML(
        '<iframe src="/ui" style="position:fixed; top:0; left:0; width:100vw; height:100vh; border:none; margin:0; padding:0; z-index:999999; background:#f4ead8;"></iframe>'
    )
    hidden_btn = gr.Button("init", visible=False)
    hidden_btn.click(fn=_zero_gpu_ready)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)