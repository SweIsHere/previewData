import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from collections import Counter
from scipy.signal import find_peaks

def detectar_y_extraer_coro(ruta_audio, salida_audio, duracion_objetivo=30, mostrar_info=True):
    """
    Detecta y extrae el coro de una canción usando análisis multi-criterio
    
    Args:
        ruta_audio (str): Ruta del archivo de audio de entrada
        salida_audio (str): Ruta donde guardar el coro extraído
        duracion_objetivo (int): Duración deseada del coro en segundos (default: 30)
        mostrar_info (bool): Si mostrar información del proceso (default: True)
    
    Returns:
        dict: Información del coro extraído o None si falló
    """
    
    def cargar_audio(ruta, sr=22050):
        y, sr = librosa.load(ruta, sr=sr)
        if mostrar_info:
            print(f"Audio cargado: duracion = {len(y)/sr:.2f} segundos")
        return y, sr

    def analizar_segmentos_completo(y, sr, ventana_s=12, hop_s=2):
        """Análisis completo: energía, repetición y características espectrales"""
        frame_length = 2048
        hop_length = 512
        
        # 1. Calcular energía (RMS)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 2. Calcular características espectrales
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # 3. Calcular tempo y beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        ventana_frames = int(ventana_s * sr / hop_length)
        hop_frames = int(hop_s * sr / hop_length)
        
        segmentos = []
        
        if mostrar_info:
            print(f"Analizando segmentos (ventana: {ventana_s}s, salto: {hop_s}s)...")
        
        for i in range(0, len(rms) - ventana_frames, hop_frames):
            inicio_frame = i
            fin_frame = i + ventana_frames
            
            # Métricas del segmento
            energia_promedio = np.mean(rms[inicio_frame:fin_frame])
            energia_varianza = np.var(rms[inicio_frame:fin_frame])
            
            # Características espectrales
            mfcc_seg = mfcc[:, inicio_frame:fin_frame]
            centroid_promedio = np.mean(spectral_centroid[inicio_frame:fin_frame])
            chroma_seg = chroma[:, inicio_frame:fin_frame]
            
            # Estabilidad tonal (coros suelen ser más estables)
            estabilidad_tonal = 1.0 / (1.0 + np.var(chroma_seg))
            
            # Tiempo del segmento
            inicio_s = librosa.frames_to_time(inicio_frame, sr=sr, hop_length=hop_length)
            fin_s = librosa.frames_to_time(fin_frame, sr=sr, hop_length=hop_length)
            
            segmento = {
                'inicio_s': inicio_s,
                'fin_s': fin_s,
                'inicio_frame': inicio_frame,
                'fin_frame': fin_frame,
                'energia_promedio': energia_promedio,
                'energia_varianza': energia_varianza,
                'centroid_promedio': centroid_promedio,
                'estabilidad_tonal': estabilidad_tonal,
                'mfcc': mfcc_seg,
                'chroma': chroma_seg
            }
            
            segmentos.append(segmento)
        
        return segmentos, rms, times, tempo

    def calcular_similitudes(segmentos, umbral_similitud=0.7):
        """Calcula similitudes entre segmentos para encontrar repeticiones"""
        n_segmentos = len(segmentos)
        similitudes = np.zeros((n_segmentos, n_segmentos))
        
        if mostrar_info:
            print("Calculando similitudes entre segmentos...")
        
        for i in range(n_segmentos):
            for j in range(i + 1, n_segmentos):
                seg1 = segmentos[i]
                seg2 = segmentos[j]
                
                # Similitud basada en MFCC
                try:
                    if seg1['mfcc'].shape[1] == seg2['mfcc'].shape[1]:
                        corr_mfcc = np.corrcoef(seg1['mfcc'].flatten(), seg2['mfcc'].flatten())[0, 1]
                        if np.isnan(corr_mfcc):
                            corr_mfcc = 0
                    else:
                        corr_mfcc = 0
                    
                    # Similitud basada en chroma (armonía)
                    if seg1['chroma'].shape[1] == seg2['chroma'].shape[1]:
                        corr_chroma = np.corrcoef(seg1['chroma'].flatten(), seg2['chroma'].flatten())[0, 1]
                        if np.isnan(corr_chroma):
                            corr_chroma = 0
                    else:
                        corr_chroma = 0
                    
                    # Similitud combinada
                    similitud = (corr_mfcc * 0.7 + corr_chroma * 0.3)
                    similitudes[i, j] = similitud
                    similitudes[j, i] = similitud
                    
                except:
                    similitudes[i, j] = 0
                    similitudes[j, i] = 0
        
        return similitudes

    def calcular_score_coro(segmentos, similitudes):
        """Calcula un score para cada segmento basado en múltiples criterios"""
        scores = []
        
        for i, seg in enumerate(segmentos):
            # 1. Score de repetición (cuántas veces aparece similar)
            repeticiones = np.sum(similitudes[i, :] > 0.6)
            score_repeticion = repeticiones / len(segmentos)
            
            # 2. Score de energía (normalizado)
            energias = [s['energia_promedio'] for s in segmentos]
            energia_normalizada = seg['energia_promedio'] / max(energias)
            
            # 3. Score de posición (coros suelen estar en el tercio medio)
            posicion_relativa = seg['inicio_s'] / segmentos[-1]['fin_s']
            if 0.25 <= posicion_relativa <= 0.75:
                score_posicion = 1.0
            elif 0.15 <= posicion_relativa <= 0.85:
                score_posicion = 0.7
            else:
                score_posicion = 0.3
            
            # 4. Score de estabilidad tonal
            estabilidades = [s['estabilidad_tonal'] for s in segmentos]
            estabilidad_normalizada = seg['estabilidad_tonal'] / max(estabilidades)
            
            # 5. Score de varianza energética (coros suelen tener varianza moderada)
            varianzas = [s['energia_varianza'] for s in segmentos]
            varianza_normalizada = 1.0 - (seg['energia_varianza'] / max(varianzas))
            
            # Score combinado con pesos
            score_total = (
                score_repeticion * 0.35 +      # Más peso a repetición
                energia_normalizada * 0.25 +   # Energía importante pero no dominante
                score_posicion * 0.15 +        # Posición en la canción
                estabilidad_normalizada * 0.15 + # Estabilidad tonal
                varianza_normalizada * 0.10     # Varianza energética
            )
            
            scores.append({
                'indice': i,
                'score_total': score_total,
                'score_repeticion': score_repeticion,
                'energia_normalizada': energia_normalizada,
                'score_posicion': score_posicion,
                'repeticiones': repeticiones,
                'inicio_s': seg['inicio_s'],
                'fin_s': seg['fin_s']
            })
        
        return sorted(scores, key=lambda x: x['score_total'], reverse=True)

    def expandir_segmento(mejor_segmento, segmentos_originales, duracion_objetivo=30):
        """Expande el mejor segmento a la duración objetivo"""
        idx = mejor_segmento['indice']
        seg_original = segmentos_originales[idx]
        
        inicio_actual = seg_original['inicio_s']
        fin_actual = seg_original['fin_s']
        duracion_actual = fin_actual - inicio_actual
        
        if mostrar_info:
            print(f"Segmento original: {inicio_actual:.2f}s - {fin_actual:.2f}s ({duracion_actual:.2f}s)")
        
        if duracion_actual >= duracion_objetivo:
            return inicio_actual, fin_actual
        
        # Expandir hacia ambos lados
        expansion_necesaria = duracion_objetivo - duracion_actual
        expansion_cada_lado = expansion_necesaria / 2
        
        nuevo_inicio = max(0, inicio_actual - expansion_cada_lado)
        nuevo_fin = fin_actual + expansion_cada_lado
        
        # Ajustar si se sale del audio
        duracion_total = segmentos_originales[-1]['fin_s']
        if nuevo_fin > duracion_total:
            nuevo_fin = duracion_total
            nuevo_inicio = max(0, nuevo_fin - duracion_objetivo)
        
        duracion_final = nuevo_fin - nuevo_inicio
        if mostrar_info:
            print(f"Segmento expandido: {nuevo_inicio:.2f}s - {nuevo_fin:.2f}s ({duracion_final:.2f}s)")
        
        return nuevo_inicio, nuevo_fin

    def exportar_coro_refinado(path, inicio_s, fin_s, salida):
        """Exporta el coro con fade in/out"""
        if mostrar_info:
            print(f"Exportando segmento: {inicio_s:.2f}s - {fin_s:.2f}s")
        
        audio = AudioSegment.from_file(path)
        fragmento = audio[inicio_s * 1000: fin_s * 1000]
        
        # Aplicar fade suave
        fade_duration = min(1000, len(fragmento) // 10)  # Max 1 segundo o 10% del fragmento
        fragmento = fragmento.fade_in(fade_duration).fade_out(fade_duration)
        
        fragmento.export(salida, format="wav")
        if mostrar_info:
            print(f"Coro refinado exportado: {salida}")

    # === PROCESO PRINCIPAL ===
    try:
        if mostrar_info:
            print("=== DETECTOR DE CORO REFINADO ===")
        
        # Verificar que el archivo existe
        if not os.path.exists(ruta_audio):
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_audio}")
        
        # Cargar audio
        y, sr = cargar_audio(ruta_audio)
        
        # Análisis completo
        segmentos, rms, times, tempo = analizar_segmentos_completo(y, sr)
        if mostrar_info:
            print(f"Detectados {len(segmentos)} segmentos para análisis")
        
        # Calcular similitudes
        similitudes = calcular_similitudes(segmentos)
        
        # Calcular scores
        scores = calcular_score_coro(segmentos, similitudes)
        
        # Mostrar top candidatos
        if mostrar_info:
            print("\nTop 3 candidatos a coro:")
            for i, score in enumerate(scores[:3]):
                seg = segmentos[score['indice']]
                print(f"  {i+1}. Tiempo: {seg['inicio_s']:.1f}s-{seg['fin_s']:.1f}s, "
                      f"Score: {score['score_total']:.3f}, "
                      f"Repeticiones: {score['repeticiones']}")
        
        # Seleccionar mejor candidato
        mejor_candidato = scores[0]
        
        # Expandir a duración objetivo
        inicio_final, fin_final = expandir_segmento(mejor_candidato, segmentos, duracion_objetivo)
        
        # Exportar
        exportar_coro_refinado(ruta_audio, inicio_final, fin_final, salida_audio)
        
        if mostrar_info:
            print("Analisis completado exitosamente!")
        
        # Retornar información del resultado
        return {
            'exito': True,
            'inicio_s': inicio_final,
            'fin_s': fin_final,
            'duracion_s': fin_final - inicio_final,
            'score': mejor_candidato['score_total'],
            'repeticiones': mejor_candidato['repeticiones'],
            'archivo_salida': salida_audio
        }
        
    except Exception as e:
        if mostrar_info:
            print(f"Error durante el procesamiento: {str(e)}")
        return {
            'exito': False,
            'error': str(e)
        }

# === EJEMPLO DE USO ===
if __name__ == "__main__":
    # Uso simple - usar audio de la carpeta audio_to_see
    ruta_entrada = os.path.join(os.path.dirname(__file__), "audio_to_see", "The_Strokes_-_Reptilia.wav")
    ruta_salida = "The_Strokes_-_Reptilia_coro.wav"
    
    resultado = detectar_y_extraer_coro(ruta_entrada, ruta_salida, duracion_objetivo=30)
    
    