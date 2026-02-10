"""
Script para exportar parámetros del StandardScaler a JSON
Esto elimina la necesidad de scikit-learn en producción
"""
import joblib
import json
import numpy as np

# Cargar scaler
scaler = joblib.load('models/scaler.pkl')

# Extraer parámetros
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'var': scaler.var_.tolist(),
    'n_features_in': int(scaler.n_features_in_),
    'feature_names': scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else None
}

# Guardar como JSON
with open('models/scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)

print("✅ Parámetros del scaler exportados a models/scaler_params.json")
print(f"   - mean: {len(scaler_params['mean'])} valores")
print(f"   - scale: {len(scaler_params['scale'])} valores")
print(f"   - features: {scaler_params['n_features_in']}")

# Verificar que funciona igual
print("\n🔍 Verificando que el escalado manual funciona igual...")
test_data = np.array([[8500, 7200, 6.5, 35.2, 4.1, 0.65, 0.45, 0.38]])

# Método 1: Con sklearn
scaled_sklearn = scaler.transform(test_data)

# Método 2: Manual con numpy
scaled_manual = (test_data - np.array(scaler_params['mean'])) / np.array(scaler_params['scale'])

# Comparar
if np.allclose(scaled_sklearn, scaled_manual):
    print("✅ El escalado manual es idéntico al de sklearn")
else:
    print("❌ Hay diferencias en el escalado")
    print(f"   sklearn: {scaled_sklearn}")
    print(f"   manual:  {scaled_manual}")
