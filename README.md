# Tarea 2 de Optimización

Este proyecto implementa un modelo de optimización entero-mixto para un problema de planificación de producción utilizando Gurobi.

## Requisitos

- Python 3.8 o superior
- Gurobi Optimizer (con licencia válida)
- make (opcional, para usar el Makefile)

## Estructura del proyecto

- `main_base.py`: Implementación del modelo de optimización
- `*.csv`: Archivos de datos de entrada para el modelo
- `I1_P1.txt`: Descripción formal del modelo matemático
- `requirements.txt`: Dependencias del proyecto
- `Makefile`: Automatización de tareas para configuración y ejecución

## Configuración rápida

El proyecto incluye un Makefile para facilitar la configuración y ejecución:

```bash
make test      # Ejecuta el modelo
```

Para activar el entorno virtual manualmente (necesario si quieres ejecutar comandos adicionales):

```bash
source venv/bin/activate
```

Para limpiar los archivos generados:

```bash
make clean
```

## Ejecución manual

Si prefieres no usar make, puedes seguir estos pasos:

```bash
# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el modelo
python main_base.py
```

## Estructura del modelo

El modelo implementa:
- Variables de decisión para producción, compra de materiales e indicadores binarios
- Restricciones de activación, balance de materiales, incompatibilidades, presupuesto y límite de productos
- Función objetivo que maximiza la utilidad neta

Para más detalles sobre la formulación matemática, consulta el archivo `I1_P1.txt`.
