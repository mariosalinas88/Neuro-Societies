# Neuro-Societies Test Suite

Suite de pruebas para verificar el funcionamiento del modelo Neuro-Societies.

## Ejecución Rápida

```bash
# Ejecutar todos los tests básicos
python tests/test_basic.py

# Ejecutar con pytest (si está instalado)
pytest tests/ -v
```

## Tests Incluidos

### test_basic.py

1. **test_model_initialization**: Verifica que el modelo se puede crear correctamente
2. **test_single_step**: Verifica que puede ejecutar un paso sin errores
3. **test_multiple_steps**: Verifica ejecución de múltiples pasos
4. **test_agent_traits**: Verifica que los agentes tienen los rasgos neurocognitivos requeridos
5. **test_metrics_collection**: Verifica que se recolectan métricas correctamente
6. **test_reproducibility**: Verifica que el mismo seed produce los mismos resultados
7. **test_population_stability**: Verifica que la población no colapsa inmediatamente

## Requisitos

```bash
pip install mesa numpy pandas networkx
```

## Niveles de Testing

### Nivel 1: Smoke Tests (5 segundos)
```bash
python tests/test_basic.py
```

### Nivel 2: Tests de Integración (30 segundos)
```bash
python run.py --steps 50
```

### Nivel 3: Tests Completos (5 minutos)
```bash
python run.py --steps 200 --enablereproduction --coalitionenabled
```

## Interpretación de Resultados

### Éxito ✓
- Todos los tests pasan
- Población se mantiene > 30% después de 20 pasos
- Métricas en rangos válidos (0-1)
- Reproducibilidad con mismo seed

### Problemas Comunes

**ImportError**: Instalar dependencias con `pip install -r requirements.txt`

**Population collapsed**: Normal si la población inicial es muy pequeña o parámetros extremos

**Metrics out of range**: Verificar que no hay divisiones por cero en el modelo

## Agregar Nuevos Tests

Crear archivos con el patrón `test_*.py` en el directorio `tests/`:

```python
def test_mi_nueva_feature():
    """Descripción del test."""
    model = SocietyModel(seed=42, population_scale="tiny")
    # ... código del test ...
    assert condicion, "mensaje de error"
    return True
```
