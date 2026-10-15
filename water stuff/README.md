# Apartment water forecast


1. `open a new terminal, use cd (folder path where the app.py is) if necessary`
2. `pip install -r requirements.txt`
2. `python verify_features.py`  -> should print `RESULT: PASS`
3. `streamlit run app.py`

`src/features.py` holds the feature logic copied from the notebook. The raw 81 MB CSV is not needed by the app.

`All this is assuming that one has set up a proper virtual environment for python so the installation won't add bloat/mess up your devices universal installation stuff, though in this case the .venv file isn't added because the full requirements installation amounted to around 400~ MB.`

`Note, the "check_by" stuff are just debug things for the cleaned dataset, they have no bearing on the app itself`

