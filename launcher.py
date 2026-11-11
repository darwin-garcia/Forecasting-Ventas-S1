from pathlib import Path
import runpy

# Carpeta donde se encuentra launcher.py
BASE_DIR = Path(__file__).resolve().parent

# Ruta real de la aplicación
APP_PATH = BASE_DIR / "app" / "app.py"

# Verificar que exista
if not APP_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró app.py en:\n{APP_PATH}"
    )

# Ejecutar app.py
runpy.run_path(
    str(APP_PATH),
    run_name="__main__"
)