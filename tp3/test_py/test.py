import os
import random
import time
import pygame
import glob
from pygame import mixer

# Inicializar pygame
pygame.init()
mixer.init()

# Configurar la pantalla
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Secuenciador Elektron-style")

# Cargar una fuente que funcione en macOS
try:
    font = pygame.font.SysFont("Arial", 24)
except:
    font = pygame.font.Font(None, 24)  # Fuente por defecto si Arial no está disponible

class Paso:
    def __init__(self, ruta_imagen=None):
        self.encendido = random.choice([True, False])
        self.probabilidad = random.uniform(0.0, 1.0)
        self.hue = random.uniform(0.0, 1.0)
        self.ruta_imagen = ruta_imagen if ruta_imagen else ""
        
    def mostrar_imagen(self):
        if not self.encendido or random.random() > self.probabilidad:
            return False
        try:
            img = pygame.image.load(self.ruta_imagen)
            # Crear una superficie para aplicar el efecto de hue
            img_surface = pygame.Surface(img.get_size(), pygame.SRCALPHA)
            img_surface.blit(img, (0, 0))
            
            # Aplicar el efecto de tono (hue)
            if self.hue > 0:
                hue_color = pygame.Color(0, 0, 0)
                hue_color.hsva = (self.hue * 360, 100, 100, 100)
                img_surface.fill(hue_color, special_flags=pygame.BLEND_RGBA_MULT)
            
            img_surface = pygame.transform.scale(img_surface, (800, 600))
            screen.blit(img_surface, (0, 0))
            return True
        except Exception as e:
            print(f"Error cargando imagen: {self.ruta_imagen}")
            print(f"Error detallado: {str(e)}")
            return False

class Secuenciador:
    def __init__(self, cantidad_pasos=16, bpm=120, carpeta_imagenes="imagenes"):
        self.cantidad_pasos = cantidad_pasos
        self.bpm = bpm
        self.carpeta_imagenes = carpeta_imagenes
        self.pasos = []
        self.imagenes_disponibles = []
        self.cargar_imagenes()
        self.aleatorizar_pasos()
        
    def cargar_imagenes(self):
        formatos = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
        for formato in formatos:
            self.imagenes_disponibles.extend(glob.glob(os.path.join(self.carpeta_imagenes, formato)))
        
        if not self.imagenes_disponibles:
            print(f"No se encontraron imágenes en la carpeta {self.carpeta_imagenes}")
            # Crear imágenes de prueba si no hay imágenes disponibles
            self.crear_imagenes_prueba()
    
    def crear_imagenes_prueba(self):
        """Crear imágenes de prueba si no hay imágenes en la carpeta"""
        if not os.path.exists(self.carpeta_imagenes):
            os.makedirs(self.carpeta_imagenes)
        
        colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, color in enumerate(colores):
            surf = pygame.Surface((400, 300))
            surf.fill(color)
            pygame.image.save(surf, os.path.join(self.carpeta_imagenes, f"prueba_{i}.png"))
        
        # Recargar las imágenes de prueba
        formatos = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
        for formato in formatos:
            self.imagenes_disponibles.extend(glob.glob(os.path.join(self.carpeta_imagenes, formato)))
    
    def aleatorizar_pasos(self):
        self.pasos = []
        for i in range(self.cantidad_pasos):
            ruta_imagen = random.choice(self.imagenes_disponibles) if self.imagenes_disponibles else None
            nuevo_paso = Paso(ruta_imagen)
            self.pasos.append(nuevo_paso)
    
    def ejecutar(self, repeticiones=8):
        # Calcular el tiempo por paso (considerando que 4 pasos = 1 tiempo completo)
        tiempo_por_paso = 60.0 / self.bpm / 4
        
        for repeticion in range(repeticiones):
            for i, paso in enumerate(self.pasos):
                inicio_tiempo = time.time()
                
                # Manejar eventos
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                
                # Limpiar pantalla (fondo negro)
                screen.fill((0, 0, 0))
                
                # Mostrar imagen si corresponde
                paso.mostrar_imagen()
                
                # Mostrar información de texto
                texto_paso = font.render(f"Paso: {i+1}/{self.cantidad_pasos}", True, (255, 255, 255))
                texto_repeticion = font.render(f"Repetición: {repeticion+1}/{repeticiones}", True, (255, 255, 255))
                texto_bpm = font.render(f"BPM: {self.bpm}", True, (255, 255, 255))
                
                screen.blit(texto_paso, (10, 10))
                screen.blit(texto_repeticion, (10, 40))
                screen.blit(texto_bpm, (10, 70))
                
                # Actualizar pantalla
                pygame.display.flip()
                
                # Controlar el tiempo de espera
                tiempo_transcurrido = time.time() - inicio_tiempo
                tiempo_restante = tiempo_por_paso - tiempo_transcurrido
                if tiempo_restante > 0:
                    time.sleep(tiempo_restante)
        
        return True

if __name__ == "__main__":
    try:
        secuenciador = Secuenciador(
            cantidad_pasos=16,
            bpm=60,
            carpeta_imagenes="imagenes"
        )
        
        # Bucle principal
        ejecutando = True
        while ejecutando:
            secuenciador.aleatorizar_pasos()
            ejecutando = secuenciador.ejecutar(repeticiones=8)
            
    except KeyboardInterrupt:
        print("Secuenciador detenido por el usuario")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        pygame.quit()