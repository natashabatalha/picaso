import os
import sys
import subprocess


def main():
    """Launch the PICASO Streamlit driver UI app."""
    script_path = os.path.join(os.path.dirname(__file__), "Home.py")
    cmd = [sys.executable, "-m", "streamlit", "run", script_path] + sys.argv[1:]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
