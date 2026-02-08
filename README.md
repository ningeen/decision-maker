# Decision Helper

A small web app for multi-criteria decision analysis (MCDA). The current implementation supports AHP and a Choix (Bradley-Terry) weighting alternative, with a weighted-sum model for option scoring and an architecture that makes it easy to add more MCDA methods later.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
python app/main.py
```

The app will start a local NiceGUI server and print the URL in the terminal.
