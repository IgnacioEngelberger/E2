.PHONY: all setup install test clean

# Variables
VENV_NAME = venv
PYTHON = python3
PIP = $(VENV_NAME)/bin/pip
PYTHON_VENV = $(VENV_NAME)/bin/python

# Crear el entorno virtual
setup:
	@echo "Creando entorno virtual..."
	@$(PYTHON) -m venv $(VENV_NAME)
	@echo "Entorno virtual creado en './$(VENV_NAME)/'"

# Activar el entorno virtual (esta regla solo imprime el comando,
# debes ejecutarlo manualmente ya que Make no puede modificar el entorno shell actual)
activate:
	@echo "Para activar el entorno virtual, ejecuta manualmente:"
	@echo "source $(VENV_NAME)/bin/activate"

# Instalar las dependencias
install: setup activate
	@echo "Instalando dependencias..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "Dependencias instaladas correctamente."

# Ejecutar el proyecto
test: install
	@echo "Ejecutando el modelo de optimización..."
	@$(PYTHON_VENV) main_base.py

# Limpiar archivos generados
clean:
	@echo "Limpiando archivos generados..."
	@rm -rf __pycache__/ $(VENV_NAME)/ *.pyc
	@echo "Limpieza completada."

help:
	@echo "Comandos disponibles:"
	@echo "  make setup     - Crea el entorno virtual"
	@echo "  make activate  - Muestra comando para activar el entorno (ejecutar manualmente)"
	@echo "  make install   - Instala dependencias en el entorno virtual"
	@echo "  make test      - Ejecuta el modelo de optimización"
	@echo "  make clean     - Elimina archivos temporales y el entorno virtual"
