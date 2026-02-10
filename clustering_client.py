
"""
Cliente Python para consumir la API de Clustering
"""
import requests
import json
from typing import Dict, List

class ClusteringAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Verifica si el servidor está disponible"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

    def predict(self, 
                ymophg_mean: float,
                ymophg_median: float,
                anosest_mean: float,
                edad_mean: float,
                totper_mean: float,
                tasa_ocupacion: float,
                tasa_pobreza: float,
                tasa_nbi: float) -> Dict:
        """
        Predice el cluster para un departamento
        """
        data = {
            "ymophg_mean": ymophg_mean,
            "ymophg_median": ymophg_median,
            "anosest_mean": anosest_mean,
            "edad_mean": edad_mean,
            "totper_mean": totper_mean,
            "tasa_ocupacion": tasa_ocupacion,
            "tasa_pobreza": tasa_pobreza,
            "tasa_nbi": tasa_nbi
        }
        response = self.session.post(f"{self.base_url}/predict", json=data)
        return response.json()

    def predict_batch(self, data_list: List[Dict]) -> Dict:
        """Predice clusters para múltiples departamentos"""
        response = self.session.post(f"{self.base_url}/predict-batch", json=data_list)
        return response.json()

    def get_info(self) -> Dict:
        """Obtiene información del modelo"""
        response = self.session.get(f"{self.base_url}/info")
        return response.json()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear cliente
    client = ClusteringAPIClient()

    # Verificar conexión
    if not client.health_check():
        print("❌ No se puede conectar al servidor")
        exit(1)

    print("✅ Conectado al servidor\n")

    # Obtener información del modelo
    info = client.get_info()
    print(f"📊 Modelo: {info['modelo']}")
    print(f"   - Clusters: {info['n_clusters']}")
    print(f"   - Características: {info['n_features']}\n")

    # Predicción simple
    print("💡 Ejemplo de predicción simple:")
    resultado = client.predict(
        ymophg_mean=8500,
        ymophg_median=7500,
        anosest_mean=6.5,
        edad_mean=35,
        totper_mean=4,
        tasa_ocupacion=0.65,
        tasa_pobreza=0.45,
        tasa_nbi=0.35
    )
    print(f"   Cluster: {resultado['cluster_name']}")
    print(f"   Confianza: {resultado['confidence']:.2%}\n")

    # Predicción por lote
    print("📦 Ejemplo de predicción por lote:")
    datos = [
        {
            "ymophg_mean": 9000, "ymophg_median": 8000, "anosest_mean": 7,
            "edad_mean": 34, "totper_mean": 3.8, "tasa_ocupacion": 0.7,
            "tasa_pobreza": 0.4, "tasa_nbi": 0.3
        },
        {
            "ymophg_mean": 5000, "ymophg_median": 4000, "anosest_mean": 4,
            "edad_mean": 37, "totper_mean": 4.5, "tasa_ocupacion": 0.5,
            "tasa_pobreza": 0.65, "tasa_nbi": 0.55
        }
    ]
    resultado_lote = client.predict_batch(datos)
    print(f"   Total procesado: {resultado_lote['total']}")
    print(f"   Resumen: {resultado_lote['summary']}")
