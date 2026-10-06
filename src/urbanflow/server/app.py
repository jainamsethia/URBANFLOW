"""The workbench server: REST + WebSocket API and the built frontend (plan M, N; simplified).

``create_app(settings)`` builds a FastAPI app with routes under ``/api/v1``:

* ``GET /health``, ``GET /meta`` (version, controllers, generators, lane metrics)
* ``GET /scenarios`` (bundled and workspace scenarios)
* ``POST/GET /sessions``, ``GET/DELETE /sessions/{id}``, ``GET /sessions/{id}/geometry``
* ``POST /compare`` (controllers x seeds batch on one scenario; paired summary)
* ``WS /ws/sessions/{id}`` (JSON control/status messages + binary UFB1 frames)

Security: loopback by default; an optional bearer token (``Authorization`` header, or the
``bearer.<token>`` WebSocket subprotocol); cross-origin WebSockets are refused; clients pick
scenarios by name only (no paths); every payload is validated with ``extra="forbid"``.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Final, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from urbanflow._version import __version__
from urbanflow.core.errors import NotFoundError, ScenarioValidationError, UrbanFlowError
from urbanflow.core.settings import AppSettings, load_settings
from urbanflow.experiments.compare import CompareRequest, run_comparison
from urbanflow.server.session import Session, SessionInfo

__all__ = ["create_app"]

API: Final = "/api/v1"
_NAME: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_STATUS: Final = {
    "ScenarioValidationError": 422,
    "ConfigError": 422,
    "NotFoundError": 404,
    "CommandError": 409,
    "MissingDependencyError": 501,
}


class SessionCreate(BaseModel):
    """``POST /sessions``: a scenario by name, or a generator with parameters."""

    model_config = ConfigDict(extra="forbid")

    scenario: str | None = Field(default=None, max_length=64)
    source: Literal["bundled", "workspace"] = "bundled"
    generator: str | None = Field(default=None, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    controller: str | None = Field(default=None, max_length=64)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    duration: float | None = Field(default=None, gt=0, le=7 * 86400)
    label: str = Field(default="", max_length=100)

    @model_validator(mode="after")
    def _one_source(self) -> SessionCreate:
        if (self.scenario is None) == (self.generator is None):
            raise ValueError('give exactly one of "scenario" or "generator"')
        if self.scenario is not None and not _NAME.match(self.scenario):
            raise ValueError("invalid scenario name")
        return self


def _is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _frontend_dir(settings: AppSettings) -> Path | None:
    if settings.frontend_dir is not None:
        return settings.frontend_dir if (settings.frontend_dir / "index.html").is_file() else None
    packaged = Path(str(resources.files("urbanflow") / "_frontend"))
    return packaged if (packaged / "index.html").is_file() else None


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Build the workbench app (no global state; one :class:`SessionManager` per app)."""
    settings = settings or load_settings()
    token = settings.api_token.get_secret_value() if settings.api_token else None
    sessions: dict[str, Session] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        reaper = asyncio.get_running_loop().create_task(_reap())
        try:
            yield
        finally:
            reaper.cancel()
            for s in list(sessions.values()):
                await s.close()
            sessions.clear()

    async def _reap() -> None:
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            for sid, s in list(sessions.items()):
                if not s.clients and now - s.last_activity > settings.session_idle_timeout_s:
                    sessions.pop(sid, None)
                    await s.close()

    app = FastAPI(
        title="UrbanFlow",
        version=__version__,
        lifespan=lifespan,
        docs_url=f"{API}/docs",
        openapi_url=f"{API}/openapi.json",
    )

    @app.exception_handler(UrbanFlowError)
    async def _problem(_req: Request, exc: UrbanFlowError) -> JSONResponse:
        status = _STATUS.get(type(exc).__name__, 400)
        body: dict[str, Any] = {
            "type": f"urn:urbanflow:problem:{type(exc).__name__}",
            "status": status,
            "detail": str(exc),
        }
        if isinstance(exc, ScenarioValidationError):
            body["issues"] = [i.to_dict() for i in exc.issues]
        return JSONResponse(body, status_code=status, media_type="application/problem+json")

    @app.middleware("http")
    async def _auth(request: Request, call_next: Any) -> Response:
        path = request.url.path
        if token and path.startswith(API) and path != f"{API}/health":
            got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(got, token):
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def _session(sid: str) -> Session:
        s = sessions.get(sid)
        if s is None:
            raise HTTPException(404, f"unknown session {sid!r}")
        return s

    # ------------------------------------------------------------------ meta
    @app.get(f"{API}/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "sessions": len(sessions)}

    @app.get(f"{API}/meta")
    def meta() -> dict[str, Any]:
        from urbanflow.scenario.generators import list_generators
        from urbanflow.signals import controller_registry
        from urbanflow.visualization import LANE_METRICS

        return {
            "version": __version__,
            "project": settings.project_name,
            "controllers": [c for c in controller_registry.names() if c != "external"],
            "generators": [
                {"name": name, "description": e.description, "schema": e.params.model_json_schema()}
                for name, e in list_generators()
            ],
            "lane_metrics": [
                {"code": int(code), "name": code.name, "label": label, "unit": unit, "domain": dom}
                for code, (label, unit, dom) in LANE_METRICS.items()
            ],
            "limits": {"max_sessions": settings.max_sessions},
            "auth_required": token is not None,
        }

    @app.get(f"{API}/scenarios")
    def scenarios() -> list[dict[str, str]]:
        from urbanflow.scenario.io import bundled_names

        out = [{"name": n, "source": "bundled"} for n in bundled_names()]
        ws = settings.scenarios_dir
        if ws.is_dir():
            out += [
                {"name": p.stem, "source": "workspace"}
                for p in sorted(ws.glob("*.json"))
                if _NAME.match(p.stem)
            ]
        return out

    # ------------------------------------------------------------------ sessions
    def _build(req: SessionCreate) -> tuple[Any, str]:
        from urbanflow import Scenario, bundled, generate

        if req.generator is not None:
            return generate(req.generator, **req.params), req.generator
        name = req.scenario or ""
        if req.source == "bundled":
            return Scenario.load(bundled(name)), name
        path = (settings.scenarios_dir / f"{name}.json").resolve()
        if not path.is_relative_to(settings.scenarios_dir.resolve()) or not path.is_file():
            raise NotFoundError(f"no workspace scenario {name!r}")
        return Scenario.load(path), name

    def _create(req: SessionCreate) -> Session:
        from urbanflow import Simulation

        scenario, name = _build(req)
        overrides: dict[str, Any] = {}
        if req.seed is not None:
            overrides["seed"] = req.seed
        if req.duration is not None:
            overrides["duration"] = req.duration
        controllers = {"*": req.controller} if req.controller else None
        sim = Simulation(scenario, controllers=controllers, **overrides)
        return Session(sim, req.label or name, name)

    @app.post(f"{API}/sessions", status_code=201)
    async def create_session(req: SessionCreate) -> SessionInfo:
        if len(sessions) >= settings.max_sessions:
            raise HTTPException(503, f"session limit reached ({settings.max_sessions})")
        session = await asyncio.to_thread(_create, req)
        sessions[session.id] = session
        session.start()
        return session.info()

    @app.get(f"{API}/sessions")
    def list_sessions() -> list[SessionInfo]:
        return [s.info() for s in sessions.values()]

    @app.get(f"{API}/sessions/{{sid}}")
    def get_session(sid: str) -> SessionInfo:
        return _session(sid).info()

    @app.delete(f"{API}/sessions/{{sid}}", status_code=204)
    async def delete_session(sid: str) -> Response:
        s = _session(sid)
        sessions.pop(sid, None)
        await s.close()
        return Response(status_code=204)

    @app.get(f"{API}/sessions/{{sid}}/geometry")
    def geometry(sid: str) -> Response:
        from urbanflow.visualization import render_geometry

        s = _session(sid)
        body = render_geometry(s.sim.network.compiled).model_dump_json()
        return Response(body, media_type="application/json")

    @app.get(f"{API}/sessions/{{sid}}/summary")
    def summary(sid: str) -> dict[str, Any]:
        from urbanflow.server.session import _finite

        s = _session(sid)
        return {"summary": _finite(s.sim.metrics.summary()), "history": _history(s)}

    @app.post(f"{API}/compare")
    async def compare(req: CompareRequest) -> dict[str, Any]:
        return await asyncio.to_thread(run_comparison, req, settings.max_workers)

    # ------------------------------------------------------------------ websocket
    @app.websocket(f"{API}/ws/sessions/{{sid}}")
    async def session_socket(ws: WebSocket, sid: str) -> None:
        origin = ws.headers.get("origin")
        host = ws.headers.get("host", "")
        allowed = {f"http://{host}", f"https://{host}", *settings.cors_origins}
        if origin is not None and origin not in allowed and not _dev_origin(origin):
            await ws.close(code=1008)
            return
        protocols = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",")]
        if token and not any(
            hmac.compare_digest(p.removeprefix("bearer."), token)
            for p in protocols
            if p.startswith("bearer.")
        ):
            await ws.accept()
            await ws.close(code=4401)
            return
        session = sessions.get(sid)
        sub = "urbanflow.v1" if "urbanflow.v1" in protocols else None
        await ws.accept(subprotocol=sub)
        if session is None:
            await ws.close(code=4404)
            return
        await session.attach(ws)
        try:
            while True:
                raw = await ws.receive_text()
                if len(raw) > 65536:
                    await ws.close(code=1009)
                    break
                await session.handle(ws, raw)
        except WebSocketDisconnect:
            pass
        finally:
            session.detach(ws)

    # ------------------------------------------------------------------ frontend
    front = _frontend_dir(settings)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        if front is None:
            return Response(
                "<h1>UrbanFlow</h1><p>The web UI is not built. Run "
                "<code>npm --prefix frontend install && npm --prefix frontend run build</code>"
                ", or use the API at <a href='/api/v1/docs'>/api/v1/docs</a>.</p>",
                media_type="text/html",
            )
        target = (front / path).resolve()
        if path and target.is_file() and target.is_relative_to(front.resolve()):
            cache = (
                "public, max-age=31536000, immutable" if "/assets/" in f"/{path}" else "no-cache"
            )
            return FileResponse(target, headers={"Cache-Control": cache})
        return FileResponse(front / "index.html", headers={"Cache-Control": "no-cache"})

    app.state.sessions = sessions
    return app


def _dev_origin(origin: str) -> bool:
    """The Vite dev server on localhost may connect (same machine, ``npm run dev``)."""
    return bool(re.match(r"^http://(localhost|127\.0\.0\.1):5173$", origin))


def _history(s: Session) -> dict[str, list[float | None]]:
    import math

    hist = s.sim.metrics.history()
    return {k: [float(x) if math.isfinite(float(x)) else None for x in v] for k, v in hist.items()}
