import argparse
import os
import threading
import webbrowser


def main():
    """Launch the PICASO Flask driver UI app."""
    parser = argparse.ArgumentParser(prog="picaso-app", description="Launch the PICASO driver UI.")
    parser.add_argument("--host", default="127.0.0.1", help="host to serve on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="port to serve on (default: 5000)")
    parser.add_argument("--refdata", default=None, help="path to the PICASO reference data (default: $picaso_refdata)")
    parser.add_argument("--debug", action="store_true", help="run Flask in debug mode with auto-reload")
    parser.add_argument("--no-browser", action="store_true", help="do not open a web browser")
    args = parser.parse_args()

    try:
        import flask, plotly  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"picaso-app needs the optional UI dependencies ({e.name} is missing). "
                         'Install them with:  pip install "picaso[app]"')

    from picaso.driver_ui.app import create_app
    app = create_app(refdata_dir=args.refdata)

    # with the debug reloader, main() runs again in a child process; only open the browser once
    if not args.no_browser and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        url = f"http://{args.host}:{args.port}/"
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()
