"""Streamlit Community Cloud entry point.

Execute the dashboard script for every Streamlit session. Importing it as a
module would cache its top-level UI code, leaving later sessions blank.
"""

from pathlib import Path
from runpy import run_path


run_path(str(Path(__file__).parent / "dashboards" / "app.py"), run_name="__main__")
