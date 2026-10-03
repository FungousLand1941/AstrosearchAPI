"""providers.shared_ssl_context honours SSL_CERT_FILE / SSL_CERT_DIR exactly as httpx does.

Regression: the cached context (added to stop reloading certifi's bundle for every owned client)
always used certifi, so the batch/ask/explain/resolver clients stopped trusting a custom corporate
CA that httpx itself picks up from ``SSL_CERT_FILE`` / ``SSL_CERT_DIR`` (``trust_env=True``).
A local TLS server whose certificate is signed by a freshly generated CA checks the real handshake.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import http.server
import ipaddress
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

import providers

x509 = pytest.importorskip("cryptography.x509")
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


def _name(common_name: str) -> Any:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _write_pki(directory: Path) -> tuple[Path, Path, Path]:
    """A self-signed CA (ca.pem) and a server certificate for 127.0.0.1/localhost signed by it."""
    now = dt.datetime.now(dt.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("AstroSearch Test Corporate CA"))
        .issuer_name(_name("AstroSearch Test Corporate CA"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                     data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                     crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    server_key = ec.generate_private_key(ec.SECP256R1())
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("localhost"))
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                                    x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                       critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    ca_pem = directory / "ca.pem"
    ca_pem.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_pem = directory / "server.pem"
    cert_pem.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    key_pem = directory / "server.key"
    key_pem.write_bytes(server_key.private_bytes(serialization.Encoding.PEM,
                                                 serialization.PrivateFormat.PKCS8,
                                                 serialization.NoEncryption()))
    return ca_pem, cert_pem, key_pem


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class _QuietServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:  # a rejected handshake is expected
        pass


@pytest.fixture(scope="module")
def tls_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path]]:
    """(https URL, CA bundle path) of a local server whose certificate only the generated CA signs."""
    ca_pem, cert_pem, key_pem = _write_pki(tmp_path_factory.mktemp("pki"))
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(cert_pem, key_pem)
    server = _QuietServer(("127.0.0.1", 0), _Handler)
    server.socket = server_ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_address[1]}/", ca_pem
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    providers.shared_ssl_context.cache_clear()
    yield
    providers.shared_ssl_context.cache_clear()


async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    async with client:
        return await client.get(url)


def _fetch(url: str, **kwargs: Any) -> httpx.Response:
    return asyncio.run(_get(providers.new_http_client(5.0, **kwargs), url))


def test_ssl_cert_file_ca_is_trusted(tls_server: tuple[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    url, ca_pem = tls_server
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_pem))
    response = _fetch(url)
    assert response.status_code == 200 and response.text == "ok"
    # the same trust httpx itself applies with trust_env (the behaviour being mirrored)
    assert asyncio.run(_get(httpx.AsyncClient(timeout=5.0), url)).status_code == 200


def test_without_ssl_cert_file_the_handshake_fails(tls_server: tuple[str, Path]) -> None:
    url, _ = tls_server
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED|certificate verify failed"):
        _fetch(url)


def test_trust_env_false_ignores_ssl_cert_file(tls_server: tuple[str, Path],
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """As httpx: ``trust_env=False`` verifies against certifi only."""
    url, ca_pem = tls_server
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_pem))
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED|certificate verify failed"):
        _fetch(url, trust_env=False)


def test_async_builder_honours_ssl_cert_file(tls_server: tuple[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    url, ca_pem = tls_server
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_pem))

    async def scenario() -> httpx.Response:
        return await _get(await providers.new_http_client_async(5.0), url)

    assert asyncio.run(scenario()).status_code == 200
    assert providers.shared_ssl_context.is_cached()


def test_caller_verify_is_passed_through(tls_server: tuple[str, Path]) -> None:
    """``verify=False`` and a caller's own context still work (no TypeError, no shared context)."""
    url, ca_pem = tls_server
    assert _fetch(url, verify=False).status_code == 200
    own = ssl.create_default_context(cafile=str(ca_pem))
    client = providers.new_http_client(5.0, verify=own)
    assert asyncio.run(_get(client, url)).status_code == 200
    assert providers.shared_ssl_context.cache_info().currsize == 0  # nothing shared was built


def test_bundle_loaded_once_per_env_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          tls_server: tuple[str, Path]) -> None:
    _, ca_pem = tls_server
    other = tmp_path / "other.pem"
    other.write_bytes(ca_pem.read_bytes())
    loads: list[dict[str, Any]] = []
    real = ssl.create_default_context

    def counting(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        loads.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", counting)

    async def build_clients(n: int) -> list[ssl.SSLContext]:
        contexts = []
        for i in range(n):
            client = (await providers.new_http_client_async(5.0) if i % 2
                      else providers.new_http_client(5.0))
            contexts.append(client._transport._pool._ssl_context)
            await client.aclose()
        return contexts

    monkeypatch.setenv("SSL_CERT_FILE", str(ca_pem))
    first = asyncio.run(build_clients(4))
    assert len(loads) == 1 and loads[0] == {"cafile": str(ca_pem)}
    assert all(ctx is first[0] for ctx in first)

    monkeypatch.setenv("SSL_CERT_FILE", str(other))
    second = asyncio.run(build_clients(3))
    assert len(loads) == 2 and loads[1] == {"cafile": str(other)}
    assert all(ctx is second[0] for ctx in second) and second[0] is not first[0]

    monkeypatch.setenv("SSL_CERT_FILE", str(ca_pem))  # back to the first value: cached, not reloaded
    assert asyncio.run(build_clients(2))[0] is first[0]
    assert len(loads) == 2

    monkeypatch.delenv("SSL_CERT_FILE")
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    asyncio.run(build_clients(2))
    assert len(loads) == 3 and loads[2] == {"capath": str(tmp_path)}

    monkeypatch.delenv("SSL_CERT_DIR")
    asyncio.run(build_clients(2))
    import certifi

    assert len(loads) == 4 and loads[3] == {"cafile": certifi.where()}


def test_ca_locations_mirror_httpx_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSL_CERT_FILE wins over SSL_CERT_DIR; empty values are ignored; trust_env=False ignores both."""
    monkeypatch.setenv("SSL_CERT_FILE", "/a/ca.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "/a/certs")
    assert providers._ssl_ca_locations() == ("/a/ca.pem", None)
    assert providers._ssl_ca_locations(trust_env=False) == (None, None)
    monkeypatch.setenv("SSL_CERT_FILE", "")
    assert providers._ssl_ca_locations() == (None, "/a/certs")
    monkeypatch.setenv("SSL_CERT_DIR", "")
    assert providers._ssl_ca_locations() == (None, None)
