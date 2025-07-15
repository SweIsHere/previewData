import os
import csv
import subprocess
import time
import shutil
import json
from pathlib import Path

# Verificar si el archivo de función existe
try:
    from CorosDETECfunc import detectar_y_extraer_coro
except ImportError:
    print("Error: No se puede importar CorosDETECfunc.py")
    exit(1)

# === Configuración de carpetas ===
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR.parent / "spotify_songs.csv"  # Buscar en el directorio padre del proyecto
OUTPUT_DIR = BASE_DIR / "Mainpreviews"
RAWDATA_DIR = BASE_DIR / "rawdata"
PROGRESS_FILE = BASE_DIR / "progreso.json"

def cargar_progreso():
    """Carga el progreso anterior si existe"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progreso = json.load(f)
                print(f"Progreso anterior encontrado: ultima fila procesada = {progreso.get('ultima_fila', 0)}")
                print(f"  Exitosas anteriores: {progreso.get('exitosas', 0)}")
                print(f"  Fallidas anteriores: {progreso.get('fallidas', 0)}")
                return progreso
        except Exception as e:
            print(f"Error cargando progreso: {e}")
    
    return {'ultima_fila': 0, 'exitosas': 0, 'fallidas': 0}

def guardar_progreso(fila_actual, exitosas, fallidas):
    """Guarda el progreso actual"""
    progreso = {
        'ultima_fila': fila_actual,
        'exitosas': exitosas,
        'fallidas': fallidas,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progreso, f, indent=2)
    except Exception as e:
        print(f"Error guardando progreso: {e}")

def verificar_dependencias():
    """Verifica que spotdl esté instalado"""
    try:
        subprocess.run(["spotdl", "--version"], 
                      capture_output=True, check=True)
        print("spotdl encontrado correctamente")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: spotdl no está instalado o no se encuentra en PATH")
        print("Instala con: pip install spotdl")
        return False

def crear_directorios():
    """Crea los directorios necesarios"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAWDATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directorios creados/verificados:")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  RawData: {RAWDATA_DIR}")

def normalizar_nombre(nombre):
    """Elimina caracteres no válidos para nombre de archivo"""
    import re
    import unicodedata
    
    # Normalizar caracteres unicode (quitar acentos y caracteres especiales)
    nombre = unicodedata.normalize('NFKD', nombre)
    nombre = ''.join(c for c in nombre if not unicodedata.combining(c))
    
    # Convertir caracteres especiales problemáticos
    nombre = nombre.replace('ł', 'l')  # ł polaca -> l
    nombre = nombre.replace('Ł', 'L')
    nombre = nombre.replace('ń', 'n')
    nombre = nombre.replace('ś', 's')
    nombre = nombre.replace('ź', 'z')
    nombre = nombre.replace('ż', 'z')
    
    # Reemplazar caracteres problemáticos
    nombre = re.sub(r'[<>:"/\\|?*]', '', nombre)
    nombre = re.sub(r'[^\w\s-]', '', nombre)
    nombre = re.sub(r'\s+', '_', nombre.strip())
    
    # Limitar longitud
    return nombre[:100] if len(nombre) > 100 else nombre

def limpiar_directorio_rawdata():
    """Elimina archivos anteriores en rawdata"""
    for archivo in RAWDATA_DIR.glob("*"):
        try:
            archivo.unlink()
        except Exception as e:
            print(f"No se pudo eliminar {archivo}: {e}")

def buscar_mp3_descargado(directorio):
    """Busca cualquier archivo de audio descargado"""
    time.sleep(2)  # Esperar que termine la descarga
    
    # Buscar archivos de audio
    archivos_audio = (list(directorio.glob("*.mp3")) + 
                     list(directorio.glob("*.m4a")) + 
                     list(directorio.glob("*.wav")))
    
    if not archivos_audio:
        return None
    
    # Devolver el más reciente
    return max(archivos_audio, key=lambda f: f.stat().st_mtime)

def safe_print(mensaje):
    """Función para imprimir de forma segura, manejando caracteres unicode"""
    try:
        print(mensaje)
    except UnicodeEncodeError:
        # Si falla, convertir a ASCII
        mensaje_safe = mensaje.encode('ascii', 'ignore').decode('ascii')
        print(mensaje_safe)

def procesar_cancion(track, artist, contador_total):
    """Procesa una canción individual"""
    safe_print(f"\n[{contador_total}] Procesando: {artist} - {track}")
    
    # Construir nombres - manejar caracteres especiales
    try:
        nombre_base = normalizar_nombre(f"{artist} - {track}")
        salida_preview = OUTPUT_DIR / f"{nombre_base}_preview.wav"
    except UnicodeError as e:
        safe_print(f"  Error con caracteres especiales: {e}")
        # Fallback: usar solo ASCII
        import re
        nombre_base = re.sub(r'[^\x00-\x7F]+', '_', f"{artist} - {track}")
        nombre_base = normalizar_nombre(nombre_base)
        salida_preview = OUTPUT_DIR / f"{nombre_base}_preview.wav"

    # Saltar si ya existe el preview
    if salida_preview.exists():
        safe_print(f"  Ya existe preview: {salida_preview.name}")
        return True

    # Limpiar directorio temporal
    limpiar_directorio_rawdata()

    # === Paso 1: Descargar canción con spotdl ===
    try:
        # Limpiar la query de caracteres problemáticos para spotdl
        artist_clean = artist.encode('ascii', 'ignore').decode('ascii')
        track_clean = track.encode('ascii', 'ignore').decode('ascii')
        query = f"{artist_clean} - {track_clean}"
        
        safe_print(f"  Descargando: spotdl \"{query}\"")
        
        # Ejecutar con encoding explícito
        resultado = subprocess.run([
            "spotdl", 
            query, 
            "--output", 
            str(RAWDATA_DIR)
        ], 
        capture_output=True, 
        text=True, 
        timeout=120,
        encoding='utf-8',
        errors='replace'
        )
        
        safe_print(f"  Return code: {resultado.returncode}")
        
        if resultado.returncode != 0:
            safe_print(f"  Error en spotdl:")
            if resultado.stderr:
                # Manejar stderr con encoding seguro
                stderr_safe = resultado.stderr.encode('ascii', 'ignore').decode('ascii')
                safe_print(f"    {stderr_safe[:300]}")
            return False
            
    except subprocess.TimeoutExpired:
        safe_print("  Timeout al descargar. Saltando...")
        return False
    except UnicodeError as e:
        safe_print(f"  Error de encoding: {e}")
        return False
    except Exception as e:
        safe_print(f"  Error al ejecutar spotdl: {e}")
        return False

    # === Paso 2: Buscar archivo descargado ===
    archivo_descargado = buscar_mp3_descargado(RAWDATA_DIR)
    
    if not archivo_descargado:
        safe_print("  No se encontro ningun archivo descargado.")
        return False

    safe_print(f"  Archivo descargado: {archivo_descargado.name}")

    # === Paso 3: Detectar y extraer coro ===
    try:
        safe_print("  Extrayendo fragmento representativo...")
        resultado = detectar_y_extraer_coro(
            str(archivo_descargado), 
            str(salida_preview), 
            duracion_objetivo=30,
            mostrar_info=False
        )
        
        if not resultado['exito']:
            safe_print(f"  Error al procesar audio: {resultado['error']}")
            return False
            
        safe_print(f"  Preview creado: {salida_preview.name}")
        safe_print(f"    Duracion: {resultado['duracion_s']:.1f}s, Score: {resultado['score']:.3f}")
        
    except Exception as e:
        safe_print(f"  Error al procesar el audio: {e}")
        return False

    # === Paso 4: Limpiar archivos temporales ===
    try:
        archivo_descargado.unlink()
        safe_print("  Archivo temporal eliminado.")
    except Exception as e:
        safe_print(f"  No se pudo eliminar el archivo temporal: {e}")

    return True

# === MAIN CON PROGRESO ===
if __name__ == "__main__":
    print("=== PROCESADOR DE DATASET SPOTIFY ===")
    
    # Verificaciones iniciales
    if not verificar_dependencias():
        exit(1)
    
    if not Path(CSV_PATH).exists():
        print(f"Error: No se encuentra el archivo CSV: {CSV_PATH}")
        exit(1)
    
    crear_directorios()
    
    # Cargar progreso anterior
    progreso = cargar_progreso()
    ultima_fila_procesada = progreso['ultima_fila']
    exitosas_previas = progreso['exitosas']
    fallidas_previas = progreso['fallidas']
    
    # Procesar CSV con encoding UTF-8
    try:
        with open(CSV_PATH, newline='', encoding='utf-8', errors='replace') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Contar total de filas para progreso
            filas = list(reader)
            total_canciones = len(filas)
            procesadas = 0
            exitosas = exitosas_previas
            fallidas = fallidas_previas
            
            print(f"\nTotal de canciones en CSV: {total_canciones}")
            
            if ultima_fila_procesada > 0:
                print(f"Continuando desde fila: {ultima_fila_procesada + 1}")
                print(f"Progreso anterior: {exitosas} exitosas, {fallidas} fallidas")
                restantes = total_canciones - ultima_fila_procesada
                print(f"Canciones restantes: {restantes}")
            else:
                print(f"Comenzando procesamiento completo")
            
            for i, row in enumerate(filas, 1):
                # Saltar filas ya procesadas
                if i <= ultima_fila_procesada:
                    continue
                
                track = row.get('track_name', '').strip()
                artist = row.get('track_artist', '').strip()
                
                if not track or not artist:
                    safe_print(f"\n[{i}] Saltando fila con datos incompletos")
                    continue
                
                procesadas += 1
                
                # Procesar con manejo de errores unicode
                try:
                    if procesar_cancion(track, artist, i):
                        exitosas += 1
                    else:
                        fallidas += 1
                except UnicodeError as e:
                    safe_print(f"  Error de encoding en {artist} - {track}: {e}")
                    fallidas += 1
                except Exception as e:
                    safe_print(f"  Error procesando {artist} - {track}: {e}")
                    fallidas += 1
                
                # Guardar progreso cada 5 canciones
                if i % 5 == 0:
                    guardar_progreso(i, exitosas, fallidas)
                
                # Progreso cada 10 canciones
                if procesadas % 10 == 0:
                    tasa_exito = (exitosas / (exitosas + fallidas)) * 100 if (exitosas + fallidas) > 0 else 0
                    print(f"\n--- Progreso: {i}/{total_canciones} filas ---")
                    print(f"    Exitosas: {exitosas}")
                    print(f"    Fallidas: {fallidas}")
                    print(f"    Tasa de exito: {tasa_exito:.1f}%")
                
                # Pausa entre canciones
                time.sleep(2)
            
            # Guardar progreso final
            guardar_progreso(total_canciones, exitosas, fallidas)
            
            print(f"\n=== RESUMEN FINAL ===")
            print(f"Total procesadas: {exitosas + fallidas}")
            print(f"Exitosas: {exitosas}")
            print(f"Fallidas: {fallidas}")
            if (exitosas + fallidas) > 0:
                tasa_final = (exitosas / (exitosas + fallidas)) * 100
                print(f"Tasa de exito: {tasa_final:.1f}%")
            
            # Limpiar archivo de progreso si terminó completamente
            if i >= total_canciones and PROGRESS_FILE.exists():
                PROGRESS_FILE.unlink()
                print("Progreso completado - archivo de progreso eliminado")
            
    except KeyboardInterrupt:
        print(f"\n\nProceso interrumpido por el usuario en fila {i}.")
        guardar_progreso(i, exitosas, fallidas)
        print("El progreso ha sido guardado.")
        print("Puedes continuar ejecutando el script nuevamente.")
    except Exception as e:
        print(f"\nError al procesar CSV: {e}")
        if 'i' in locals():
            guardar_progreso(i, exitosas, fallidas)
            print("Progreso guardado antes del error.")