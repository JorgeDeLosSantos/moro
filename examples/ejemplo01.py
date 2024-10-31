import numpy as np
import matplotlib.pyplot as plt

# Longitudes de las barras
a, b, c, d, e, f, g, h, i, j = 38.0, 41.5, 39.3, 40.1, 55.8, 39.4, 36.7, 65.7, 49.0, 50.0

# Función para calcular coordenadas
def calculate_leg_positions(theta):
    # Convertimos el ángulo a radianes
    theta = np.radians(theta)
    
    # Posiciones fijas
    A = np.array([0, 0])  # Punto de rotación fijo
    B = np.array([0, -a])  # Punto fijo en el suelo
    
    # Calcula posición de punto C usando el ángulo theta
    C = np.array([a * np.cos(theta), a * np.sin(theta)])
    
    # Calcula otros puntos aplicando ley de cosenos o senos según sea necesario
    # Por ejemplo, posición de D conectada a C y B
    # Aquí puedes completar los cálculos geométricos para cada punto
    
    # Ejemplo para calcular el siguiente punto, debes añadir los otros:
    # Asumiendo D está en (x_D, y_D) en relación a los otros puntos:
    # (Agrega la resolución de cada punto con las leyes trigonométricas)
    
    # Retorna posiciones de los puntos
    return A, B, C  # Completa con el resto de puntos que necesites

# Genera coordenadas para varios ángulos
angles = np.linspace(0, 360, 100)  # 100 pasos
coordinates = [calculate_leg_positions(angle) for angle in angles]

# Opcional: Visualización
plt.figure()
for coord in coordinates:
    plt.plot(*zip(*coord), marker='o')
plt.show()
