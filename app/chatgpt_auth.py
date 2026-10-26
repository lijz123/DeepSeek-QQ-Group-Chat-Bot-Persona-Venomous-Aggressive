"""Official ChatGPT-plan OAuth for this application (Windows local storage).

This module never reads Codex credentials, browser cookies, or API keys. Login
does not make an inference request. Tokens remain outside the source directory
and are encrypted using Windows CurrentUser DPAPI.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import ctypes
import hashlib
import hmac
import html
import json
import math
import os
import re
import secrets
import subprocess
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from ctypes import wintypes
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import getproxies

from project_paths import PROJECT_ROOT

import aiohttp
from aiohttp import web
import jwt

APP = "QQ Roast Bot"
ISSUER = "https://auth.openai.com"
AUTH = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
REQUIRED = {"chatgpt.tokens.use.direct", "resource.invoke"}
ALGORITHMS = {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}
TERMINAL_REFRESH_ERRORS = {"invalid_grant", "invalid_refresh_token", "token_expired",
                           "refresh_token_expired", "refresh_token_invalidated",
                           "refresh_token_reused"}
_LOCAL_LOCKS: dict[str, asyncio.Lock] = {}


class PlanAuthError(Exception):
    """An intentionally sanitized error: do not include server bodies/tokens."""
    def __init__(self, code: str):
        self.code = code if re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", str(code)) else "AUTH_FAILED"
        super().__init__(self.code)


def default_profile_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise PlanAuthError("WINDOWS_LOCALAPPDATA_REQUIRED")
    return Path(base) / "QQRoastBot" / "chatgpt-plan"


def _reject_links(path: Path) -> None:
    for p in (path, *path.parents):
        if p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()):
            raise PlanAuthError("PROFILE_LINK_NOT_ALLOWED")


def _atomic_bytes(path: Path, data: bytes) -> None:
    _reject_links(path)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def _read_json(path: Path) -> dict:
    _reject_links(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError()
        return data
    except (OSError, ValueError):
        raise PlanAuthError("PROFILE_METADATA_INVALID") from None


def _private_acl(path: Path) -> None:
    if os.name != "nt":
        raise PlanAuthError("WINDOWS_PROTECTED_STORAGE_REQUIRED")
    # A fresh explicit DACL, not merely removal of inherited permissions.
    literal = str(path).replace("'", "''")
    script = (
        "$ErrorActionPreference='Stop';"
        "$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User;"
        "$system=[Security.Principal.SecurityIdentifier]::new('S-1-5-18');"
        f"$p='{literal}';"
        "$a=[Security.AccessControl.DirectorySecurity]::new();"
        "$a.SetOwner($sid);$a.SetAccessRuleProtection($true,$false);"
        "foreach($s in @($sid,$system)){$a.AddAccessRule("
        "[Security.AccessControl.FileSystemAccessRule]::new($s,'FullControl',"
        "'ContainerInherit,ObjectInherit','None','Allow'))};"
        # Use the Windows PowerShell .NET ACL API directly: Set-Acl module
        # auto-loading can fail when launched from a newer PowerShell host.
        "[IO.Directory]::SetAccessControl($p,$a);"
        "$verified=[IO.Directory]::GetAccessControl($p);"
        "if(-not $verified.AreAccessRulesProtected){throw 'DACL not protected'};"
        "$rules=$verified.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier]);"
        "if($rules.Count -ne 2 -or $verified.GetOwner([Security.Principal.SecurityIdentifier]).Value -ne $sid.Value){throw 'ACL owner/count mismatch'};"
        "foreach($r in $rules){if($r.IdentityReference.Value -notin @($sid.Value,$system.Value) "
        "-or $r.AccessControlType -ne 'Allow' -or $r.FileSystemRights -ne 'FullControl'){throw 'Unexpected ACL entry'}};"
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    try:
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                capture_output=True, timeout=20,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise PlanAuthError("PROFILE_ACL_FAILED")
    except (OSError, subprocess.TimeoutExpired):
        raise PlanAuthError("PROFILE_ACL_FAILED") from None


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise PlanAuthError("WINDOWS_PROTECTED_STORAGE_REQUIRED")
    buf = ctypes.create_string_buffer(data)
    source = _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    output = _Blob()
    if decrypt:
        method = ctypes.windll.crypt32.CryptUnprotectData
        method.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
        arguments = (ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output))
    else:
        method = ctypes.windll.crypt32.CryptProtectData
        method.argtypes = [ctypes.POINTER(_Blob), wintypes.LPCWSTR, ctypes.c_void_p,
                           ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
        arguments = (ctypes.byref(source), APP, None, None, None, 1, ctypes.byref(output))
    method.restype = wintypes.BOOL
    if not method(*arguments):
        raise PlanAuthError("PROTECTED_STORAGE_FAILED")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        free = ctypes.windll.kernel32.LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        free(output.pbData)


def validate_record(record: dict) -> dict:
    if not isinstance(record, dict) or record.get("issuer") != ISSUER:
        raise PlanAuthError("PROFILE_ISSUER_INVALID")
    if record.get("app", APP) != APP:
        raise PlanAuthError("PROFILE_APPLICATION_INVALID")
    client = record.get("client_id")
    if not isinstance(client, str) or client == "dynamic_agent_client" or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", client):
        raise PlanAuthError("PROFILE_CLIENT_INVALID")
    for field in ("subject", "access_token", "refresh_token", "ext_agent_host_id"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise PlanAuthError("PROFILE_FIELD_INVALID_" + field.upper())
    scopes = record.get("scopes")
    if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes) or not REQUIRED <= set(scopes):
        raise PlanAuthError("PLAN_PERMISSION_NOT_GRANTED")
    for name in ("saved_at", "expires_in"):
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise PlanAuthError("PROFILE_EXPIRY_INVALID")
    earliest = record.get("earliest_refresh_at")
    if earliest is not None and (isinstance(earliest, bool) or not isinstance(earliest, (int, float)) or not math.isfinite(earliest) or earliest < 0):
        raise PlanAuthError("PROFILE_REFRESH_TIME_INVALID")
    return record


def encrypt_record(record: dict) -> bytes:
    validate_record(record)
    return _dpapi(json.dumps(record, ensure_ascii=False).encode("utf-8"), decrypt=False)


def decrypt_record(blob: bytes) -> dict:
    try:
        value = json.loads(_dpapi(blob, decrypt=True).decode("utf-8"))
    except PlanAuthError:
        raise
    except (ValueError, UnicodeError):
        raise PlanAuthError("PROFILE_DECRYPTED_DATA_INVALID") from None
    return validate_record(value)


def validate_id_token(token: str, jwks: dict, client_id: str, nonce: str | None = None) -> dict:
    try:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        if algorithm not in ALGORITHMS:
            raise PlanAuthError("UNSAFE_ID_TOKEN_ALGORITHM")
        keys = [key for key in jwks.get("keys", []) if key.get("kid") == header.get("kid")
                and key.get("use", "sig") == "sig" and key.get("alg", algorithm) == algorithm]
        if len(keys) != 1:
            raise PlanAuthError("SIGNING_KEY_NOT_FOUND")
        key = jwt.PyJWK.from_dict(keys[0]).key
        required = ["exp", "iat", "iss", "aud", "sub"] + (["nonce"] if nonce is not None else [])
        claims = jwt.decode(token, key, algorithms=[algorithm], audience=client_id, issuer=ISSUER,
                            leeway=30, options={"require": required})
        if not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise PlanAuthError("MISSING_VERIFIED_SUBJECT")
        if nonce is not None:
            observed = claims.get("nonce")
            if not isinstance(observed, str) or not observed.isascii() or not hmac.compare_digest(observed, nonce):
                raise PlanAuthError("NONCE_MISMATCH")
        audience = claims["aud"]
        if ((isinstance(audience, list) and len(audience) > 1 and claims.get("azp") != client_id)
                or ("azp" in claims and claims["azp"] != client_id)):
            raise PlanAuthError("AUTHORIZED_PARTY_MISMATCH")
        return claims
    except PlanAuthError:
        raise
    except Exception:
        raise PlanAuthError("ID_TOKEN_VALIDATION_FAILED") from None


def _safe_error(data: dict) -> str:
    error = data.get("error", {}) if isinstance(data, dict) else {}
    code = (error.get("code") or error.get("type")) if isinstance(error, dict) else error
    return code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", code) else "HTTP_REQUEST_FAILED"


async def _json_response(response, label: str) -> dict:
    try:
        data = await response.json(content_type=None)
    except Exception:
        raise PlanAuthError(label + "_NON_JSON") from None
    if not isinstance(data, dict):
        raise PlanAuthError(label + "_INVALID_RESPONSE")
    return data


async def _public_keys(session) -> dict:
    proxy = getproxies().get("https")
    async with session.get(ISSUER + "/.well-known/openid-configuration", proxy=proxy, allow_redirects=False,
                           timeout=aiohttp.ClientTimeout(total=40)) as response:
        if response.status != 200:
            raise PlanAuthError("DISCOVERY_HTTP_" + str(response.status))
        discovery = await _json_response(response, "DISCOVERY")
    if discovery.get("issuer") != ISSUER:
        raise PlanAuthError("DISCOVERY_ISSUER_MISMATCH")
    uri = discovery.get("jwks_uri", "")
    parsed = urlparse(uri)
    if parsed.scheme != "https" or parsed.hostname != "auth.openai.com" or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise PlanAuthError("UNEXPECTED_JWKS_ENDPOINT")
    async with session.get(uri, proxy=proxy, allow_redirects=False, timeout=aiohttp.ClientTimeout(total=40)) as response:
        if response.status != 200:
            raise PlanAuthError("JWKS_HTTP_" + str(response.status))
        keys = await _json_response(response, "JWKS")
    if not isinstance(keys.get("keys"), list) or not keys["keys"]:
        raise PlanAuthError("EMPTY_PUBLIC_JWKS")
    return keys


class PlanProfile:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(os.path.abspath(os.path.expanduser(str(path or default_profile_dir()))))
        self.identity_key: tuple[str, str, str] | None = None
        self._prepared = False

    def _prepare(self):
        if self._prepared:
            return
        _reject_links(self.path)
        if self.path == PROJECT_ROOT or PROJECT_ROOT in self.path.parents:
            raise PlanAuthError("PROFILE_MUST_BE_OUTSIDE_SOURCE")
        if self.path.exists():
            if not self.path.is_dir():
                raise PlanAuthError("PROFILE_DIRECTORY_REQUIRED")
            for entry in self.path.iterdir():
                _reject_links(entry)
                known = entry.name in {"credentials.dpapi", "registration.json", "profile.lock", "provider-diagnostics.jsonl"}
                temporary = re.fullmatch(r"(?:credentials\.dpapi|registration\.json)\.[0-9a-f]{16}\.tmp", entry.name)
                if not entry.is_file() or not (known or temporary):
                    raise PlanAuthError("PROFILE_DIRECTORY_NOT_DEDICATED")
        self.path.mkdir(parents=True, exist_ok=True)
        _private_acl(self.path)
        self._prepared = True

    def exists(self) -> bool:
        _reject_links(self.path / "credentials.dpapi")
        return (self.path / "credentials.dpapi").is_file()

    def _read(self) -> dict:
        _reject_links(self.path / "credentials.dpapi")
        try:
            return decrypt_record((self.path / "credentials.dpapi").read_bytes())
        except OSError:
            raise PlanAuthError("SIGN_IN_REQUIRED") from None

    def _write(self, record: dict) -> None:
        self._prepare()
        validate_record(record)
        _atomic_bytes(self.path / "credentials.dpapi", encrypt_record(record))
        _atomic_json(self.path / "registration.json", {
            "app": APP, "client_id": record["client_id"], "ext_agent_host_id": record["ext_agent_host_id"],
            "identity_validated": True,
            "subject_hash": hashlib.sha256(record["subject"].encode("utf-8")).hexdigest(),
        })

    @asynccontextmanager
    async def _process_lock(self, timeout: float = 20):
        self._prepare()
        lock_path = self.path / "profile.lock"
        _reject_links(lock_path)
        lock = lock_path.open("a+b")
        lock.seek(0)
        try:
            import msvcrt
            deadline = time.monotonic() + timeout
            while True:
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise PlanAuthError("PROFILE_BUSY") from None
                    await asyncio.sleep(0.1)
            try:
                # Windows byte-range locks may cover the byte beyond EOF. Seed
                # a fresh lock file only AFTER owning it, never through another
                # process's already-locked byte.
                if os.fstat(lock.fileno()).st_size == 0:
                    lock.write(b"0")
                    lock.flush()
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            lock.close()

    async def access_token(self, session) -> str:
        local = _LOCAL_LOCKS.setdefault(str(self.path).casefold(), asyncio.Lock())
        async with local, self._process_lock():
            record = self._read()  # Read only AFTER acquiring both rotating-token locks.
            self.identity_key = (record["issuer"], record["client_id"], record["subject"])
            now = time.time()
            expires_at = record["saved_at"] + record["expires_in"]
            earliest = record.get("earliest_refresh_at") or 0
            if now < expires_at - 60 or (now < earliest and now < expires_at):
                return record["access_token"]
            if now < earliest:
                raise PlanAuthError("REFRESH_NOT_YET_ALLOWED")
            try:
                # Fetch validation keys BEFORE rotating the refresh token. A failed
                # public-key fetch must not strand a newly issued replacement.
                jwks = await _public_keys(session)
                async with session.post(TOKEN, data={"grant_type": "refresh_token",
                        "client_id": record["client_id"], "refresh_token": record["refresh_token"],
                        "resource": RESOURCE}, proxy=getproxies().get("https"), allow_redirects=False,
                        timeout=aiohttp.ClientTimeout(total=40)) as response:
                    data = await _json_response(response, "REFRESH")
                    if response.status != 200:
                        code = _safe_error(data)
                        if code in TERMINAL_REFRESH_ERRORS:
                            _reject_links(self.path / "credentials.dpapi")
                            (self.path / "credentials.dpapi").unlink(missing_ok=True)
                            self.identity_key = None
                            raise PlanAuthError("SIGN_IN_REQUIRED_" + code)
                        raise PlanAuthError("REFRESH_" + code)
                if data.get("token_type", "").lower() != "bearer" or not data.get("access_token") or not data.get("refresh_token"):
                    raise PlanAuthError("REFRESH_INVALID_TOKEN_RESPONSE")
                scopes = data.get("scope")
                if scopes is not None:
                    if not isinstance(scopes, str) or not REQUIRED <= set(scopes.split()):
                        raise PlanAuthError("PLAN_PERMISSION_NOT_GRANTED")
                    record["scopes"] = scopes.split()
                if data.get("id_token") is not None:
                    claims = validate_id_token(data["id_token"], jwks, record["client_id"])
                    if claims["sub"] != record["subject"]:
                        raise PlanAuthError("REFRESH_IDENTITY_MISMATCH")
                    record["id_token"] = data["id_token"]
                record.update(access_token=data["access_token"], refresh_token=data["refresh_token"],
                              saved_at=time.time(), expires_in=data.get("expires_in"),
                              earliest_refresh_at=data.get("earliest_refresh_at"))
                self._write(record)
                return record["access_token"]
            except PlanAuthError:
                raise
            except Exception:
                raise PlanAuthError("REFRESH_NETWORK_FAILED_NO_AUTOMATIC_RETRY") from None


def pkce_challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode().rstrip("=")


def validate_callback(query: dict, expected_state: str, existing_client: str | None) -> dict:
    state = query.get("state", "")
    if not isinstance(state, str) or not state.isascii() or not hmac.compare_digest(state, expected_state):
        raise PlanAuthError("STATE_MISMATCH")
    if query.get("error"):
        raise PlanAuthError("AUTHORIZATION_DENIED")
    code = query.get("code")
    client = query.get("client_id") or existing_client
    if not isinstance(code, str) or not code or len(code) > 4096:
        raise PlanAuthError("MISSING_CODE")
    if not isinstance(client, str) or client == "dynamic_agent_client" or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", client):
        raise PlanAuthError("MISSING_ISSUED_CLIENT_ID")
    if existing_client and client != existing_client:
        raise PlanAuthError("CLIENT_ID_MISMATCH")
    return {"code": code, "client_id": client}


async def sign_in(profile_path: str | Path | None = None) -> dict:
    """User-operated official browser authorization; ZERO inference requests."""
    profile = PlanProfile(profile_path)
    local = _LOCAL_LOCKS.setdefault(str(profile.path).casefold(), asyncio.Lock())
    async with local, profile._process_lock(timeout=1):
        reg_path = profile.path / "registration.json"
        reg = _read_json(reg_path) if reg_path.exists() else {}
        if reg and reg.get("app") != APP:
            raise PlanAuthError("PROFILE_APPLICATION_INVALID")
        existing_client = reg.get("client_id")
        if existing_client and (existing_client == "dynamic_agent_client" or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", existing_client)):
            raise PlanAuthError("PROFILE_CLIENT_INVALID")
        host = reg.get("ext_agent_host_id") or "urn:uuid:" + str(uuid.uuid4())
        reg.update(app=APP, ext_agent_host_id=host)
        _atomic_json(reg_path, reg)
        verifier, state, nonce = secrets.token_urlsafe(48), secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        callback = asyncio.get_running_loop().create_future()
        port, authorize_url = 0, ""
        headers = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
                   "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'"}
        @web.middleware
        async def guard(request, handler):
            peer = request.transport.get_extra_info("peername") if request.transport else None
            if request.host != f"127.0.0.1:{port}" or not peer or peer[0] != "127.0.0.1":
                raise web.HTTPBadRequest(text="Invalid local request")
            return await handler(request)
        async def landing(request):
            return web.Response(text='<meta charset="utf-8"><h2>QQ Roast Bot</h2><p>登录仅在 auth.openai.com 官方页面进行。</p><a href="' +
                                html.escape(authorize_url, quote=True) + '">Continue with ChatGPT</a>', content_type="text/html", headers=headers)
        async def receive(request):
            if any(len(request.query.getall(k)) != 1 for k in request.query):
                raise web.HTTPBadRequest(text="Duplicate parameter")
            try:
                result = validate_callback(dict(request.query), state, existing_client)
            except PlanAuthError as error:
                if error.code == "AUTHORIZATION_DENIED" and not callback.done():
                    callback.set_exception(error)
                return web.Response(text="Authorization declined or invalid. Return to the application.", status=400, headers=headers)
            if not callback.done():
                callback.set_result(result)
            return web.Response(status=303, headers={**headers, "Location": "/result"})
        async def finished(request):
            return web.Response(text='<meta charset="utf-8"><p>已收到回调，正在验证。请返回程序查看结果。</p>', content_type="text/html", headers=headers)
        app = web.Application(middlewares=[guard], client_max_size=16384)
        app.router.add_get("/", landing)
        app.router.add_get("/auth/callback", receive)
        app.router.add_get("/result", finished)
        runner = web.AppRunner(app, access_log=None)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40), cookie_jar=aiohttp.DummyCookieJar()) as session:
                await _public_keys(session)  # Fail before asking for consent if validation cannot work.
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                port = site._server.sockets[0].getsockname()[1]
                redirect = f"http://127.0.0.1:{port}/auth/callback"
                params = {"client_id": existing_client or "dynamic_agent_client", "ext_agent_host_id": host,
                          "response_type": "code", "redirect_uri": redirect,
                          "scope": "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
                          "resource": RESOURCE, "state": state, "nonce": nonce,
                          "code_challenge_method": "S256", "code_challenge": pkce_challenge(verifier)}
                if not existing_client:
                    params["agent_name_hint"] = APP
                authorize_url = AUTH + "?" + urlencode(params)
                print(f"请在官方页面完成登录和授权。若浏览器没有弹出，请打开 http://127.0.0.1:{port}/", flush=True)
                await asyncio.to_thread(webbrowser.open, authorize_url, new=2, autoraise=True)
                received = await asyncio.wait_for(callback, timeout=1200)
                client = received["client_id"]
                reg.update(client_id=client)
                _atomic_json(reg_path, reg)
                async with session.post(TOKEN, data={"grant_type": "authorization_code", "client_id": client,
                        "code": received["code"], "code_verifier": verifier, "redirect_uri": redirect,
                        "resource": RESOURCE}, proxy=getproxies().get("https"), allow_redirects=False) as response:
                    data = await _json_response(response, "TOKEN")
                    if response.status != 200:
                        raise PlanAuthError("TOKEN_" + _safe_error(data))
                if data.get("token_type", "").lower() != "bearer" or not isinstance(data.get("id_token"), str):
                    raise PlanAuthError("INVALID_TOKEN_RESPONSE")
                claims = validate_id_token(data["id_token"], await _public_keys(session), client, nonce)
                subject_hash = hashlib.sha256(claims["sub"].encode("utf-8")).hexdigest()
                if reg.get("subject_hash") and not hmac.compare_digest(reg["subject_hash"], subject_hash):
                    raise PlanAuthError("RETURNING_IDENTITY_MISMATCH")
                scopes = data.get("scope")
                if not isinstance(scopes, str) or not REQUIRED <= set(scopes.split()):
                    raise PlanAuthError("PLAN_PERMISSION_NOT_GRANTED")
                record = {"app": APP, "issuer": ISSUER, "client_id": client, "ext_agent_host_id": host,
                          "subject": claims["sub"], "access_token": data.get("access_token"),
                          "refresh_token": data.get("refresh_token"), "id_token": data["id_token"],
                          "scopes": scopes.split(), "saved_at": time.time(), "expires_in": data.get("expires_in"),
                          "earliest_refresh_at": data.get("earliest_refresh_at")}
                profile._write(record)
                return {"signed_in": True, "plan_permission": True, "inference_requests": 0}
        except PlanAuthError:
            raise
        except asyncio.TimeoutError:
            raise PlanAuthError("AUTHORIZATION_TIMEOUT_NO_RETRY") from None
        except Exception:
            raise PlanAuthError("AUTHORIZATION_FAILED_NO_AUTOMATIC_RETRY") from None
        finally:
            await runner.cleanup()


async def import_verified_profile_async(source_dir: str | Path, target_dir: str | Path | None = None) -> dict:
    """Explicit migration of THIS app's previously verified helper profile only."""
    source = Path(os.path.abspath(str(source_dir)))
    _reject_links(source)
    reg = _read_json(source / "registration.json")
    status = _read_json(source / "status.json")
    if reg.get("app") != APP or reg.get("identity_validated") is not True or status.get("identity_verified") is not True or status.get("plan_permission_verified") is not True:
        raise PlanAuthError("SOURCE_IS_NOT_THIS_APPLICATION_VERIFIED_PROFILE")
    _reject_links(source / "credentials.dpapi")
    try:
        record = decrypt_record((source / "credentials.dpapi").read_bytes())
    except OSError:
        raise PlanAuthError("SOURCE_PROFILE_MISSING") from None
    if reg.get("client_id") != record["client_id"] or reg.get("ext_agent_host_id") != record["ext_agent_host_id"]:
        raise PlanAuthError("SOURCE_PROFILE_IDENTITY_MISMATCH")
    target = PlanProfile(target_dir)
    local = _LOCAL_LOCKS.setdefault(str(target.path).casefold(), asyncio.Lock())
    async with local, target._process_lock(timeout=1):
        if target.exists():
            old = target._read()
            if (old["issuer"], old["client_id"], old["subject"]) != (record["issuer"], record["client_id"], record["subject"]):
                raise PlanAuthError("TARGET_HAS_DIFFERENT_ACCOUNT")
            raise PlanAuthError("TARGET_ALREADY_HAS_PROFILE_USE_EXISTING_SESSION")
        reg_path = target.path / "registration.json"
        if reg_path.exists():
            prior = _read_json(reg_path)
            identity_hash = hashlib.sha256(record["subject"].encode("utf-8")).hexdigest()
            if prior.get("client_id") not in (None, record["client_id"]) or prior.get("subject_hash") not in (None, identity_hash):
                raise PlanAuthError("TARGET_HAS_DIFFERENT_ACCOUNT")
        record["app"] = APP
        target._write(record)
    return {"imported": True, "plan_permission": True, "inference_requests": 0}


def import_verified_profile(source_dir: str | Path, target_dir: str | Path | None = None) -> dict:
    """Synchronous CLI-friendly entry; async callers use the async variant."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(import_verified_profile_async(source_dir, target_dir))
    raise PlanAuthError("USE_ASYNC_IMPORT_WITH_RUNNING_LOOP")


async def _cli(args):
    profile = PlanProfile(args.profile)
    if args.login:
        return await sign_in(args.profile)
    if args.import_verified_profile:
        return await import_verified_profile_async(args.import_verified_profile, args.profile)
    if args.logout:
        local = _LOCAL_LOCKS.setdefault(str(profile.path).casefold(), asyncio.Lock())
        async with local, profile._process_lock():
            _reject_links(profile.path / "credentials.dpapi")
            (profile.path / "credentials.dpapi").unlink(missing_ok=True)
        return {"signed_in": False, "local_logout_only": True,
                "notice": "远端撤销未确认；请在 ChatGPT 设置中断开 QQ Roast Bot 的授权。"}
    if not profile.exists():
        return {"signed_in": False, "profile": str(profile.path)}
    value = profile._read()
    return {"signed_in": True, "plan_permission": True,
            "account_label": "ChatGPT-" + hashlib.sha256(value["subject"].encode()).hexdigest()[:8],
            "access_token_expired": time.time() >= value["saved_at"] + value["expires_in"],
            "profile": str(profile.path), "network_requests": 0}


def cli_main() -> int:
    parser = argparse.ArgumentParser(description="QQ Roast Bot — official ChatGPT-plan sign-in; no inference")
    parser.add_argument("--profile", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--login", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--logout", action="store_true")
    group.add_argument("--import-verified-profile", metavar="SOURCE_DIRECTORY")
    args = parser.parse_args()
    try:
        print(json.dumps(asyncio.run(_cli(args)), ensure_ascii=False))
        return 0
    except PlanAuthError as error:
        print(json.dumps({"ok": False, "error_code": error.code}, ensure_ascii=False))
        return 1
    except KeyboardInterrupt:
        print('{"ok":false,"error_code":"CANCELLED"}')
        return 130
    except Exception:
        print('{"ok":false,"error_code":"LOCAL_AUTH_OPERATION_FAILED"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
