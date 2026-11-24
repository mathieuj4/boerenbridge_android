"""Entry point used by the Android app: starts the game server inside the app and returns its port."""
import os
import threading

_port = None


def start(data_dir):
    global _port
    if _port is not None:
        return _port
    os.environ["BB_DATA"] = data_dir
    import play
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), play.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _port = srv.server_address[1]
    return _port
