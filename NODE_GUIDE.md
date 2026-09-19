# MiniMax H3 · Two Pass

Reinicia ComfyUI y recarga el navegador. Busca **MiniMax H3 · Two Pass** en `MiniMax H3/sampling`.

Conecta `model_pass1`, `positive`, `negative` y el `latent` audiovisual inicial. La única salida, `latent`, va a tus decodificadores de vídeo/audio y al guardado final. `model_pass2` es opcional: si no se conecta, ambas pasadas usan `model_pass1`.

## Etapas

Con Pass 2 activo, `total_steps=8` y `split_step=4` ejecutan los intervalos 0–4 y 4–8. Cada pasada tiene semilla, CFG, sampler y scheduler propios. Si desactivas Pass 2, Pass 1 completa todos los pasos y termina sin ruido restante. El upscale se puede activar independientemente.

Los valores iniciales reproducen el esquema de muestreo del workflow: LCM/simple, CFG 1, Pass 1 sin ruido sobrante y Pass 2 añadiendo ruido. Las semillas se configuran por separado. Modelos, LoRA y sigma shift se preparan antes de entrar al nodo.

## Pass 2 por chunks temporales

En la pestaña **Pass 2**, selecciona `pass2_sampling_mode = temporal`. El modo `full` mantiene el comportamiento anterior.

- `pass2_chunking_mode = auto (chunk count)`: expone sólo `pass2_chunk_count` y calcula el tamaño de ventana y el solapamiento según la duración real del latente.
- `pass2_chunk_count = 2`: para un vídeo de 243 fotogramas (unos 10 segundos), calcula `chunk=141` y `overlap=22`.
- `pass2_chunking_mode = manual (frames)`: expone `pass2_chunk_frames` y `pass2_overlap_frames` para controlar directamente la memoria y las transiciones. Los dos modos son excluyentes.

El modo automático usa como objetivo unos 22 fotogramas de solapamiento y redondea el tamaño a la cuadrícula de H3 (5 + 17k), de forma que el número solicitado de ventanas cubra el vídeo completo. Los inicios mantienen la fase temporal del latente; la última ventana puede ser más corta para conservar el solapamiento sin truncar el vídeo.

Pass 2 ejecuta una sola trayectoria de muestreo (`split_step → total_steps`). En cada evaluación del modelo, todas las ventanas reciben sus posiciones del mismo estado global; sus predicciones se combinan con pesos normalizados en el overlap antes de avanzar el sampler. Se conserva así el historial del solver y el ruido estocástico global. No se congelan overlaps ni se ensamblan clips terminados mediante cortes o crossfade. El estado global ocupa memoria durante todo el muestreo, mientras el modelo de difusión procesa una ventana a la vez.

Con chunks de 56/22, las ventanas avanzan 34 fotogramas; un vídeo de 362 fotogramas utiliza 10 ventanas por evaluación. El indicador de chunks vuelve al primero en cada evaluación; la barra de pasos mide el progreso global. La calidad visual requiere validación con el checkpoint real.

**Los chunks no regeneran el audio.** Se proporciona al modelo el tramo correspondiente con máscara de denoising cero: conservan exactamente el audio recibido, original o refinado según `pass2_audio_mode`. `pass1_return_with_leftover_noise` debe estar desactivado para trabajar con latentes limpios. Las guías de vídeo/audio ancladas en el tiempo se recortan y reposicionan para cada ventana; las referencias independientes de identidad se conservan.

### Refinamiento independiente del audio

En Pass 2 temporal, `pass2_audio_mode` permite elegir:

- `preserve` (predeterminado): conserva el audio de Pass 1, sin coste adicional. Mantiene el comportamiento de los workflows existentes.
- `refine`: refina la pista completa antes del upscale, usando el vídeo de Pass 1 congelado como contexto. Después, los chunks de vídeo conservan ese audio refinado.

Empieza con `pass2_audio_steps = 8` y `pass2_audio_start_step = 4`: ejecuta los últimos cuatro pasos de un calendario de ocho. Este calendario es independiente de `total_steps` y `split_step`; no cambia los ajustes de chunks del vídeo. Comparte modelo, seed, CFG, sampler, scheduler y `pass2_add_noise` de Pass 2. Si no conectas `model_pass2`, utiliza `model_pass1`.

El recorrido es Pass 1 → preview opcional → refinamiento de audio → upscale opcional → chunks de vídeo. El preview sigue mostrando el vídeo y audio originales de Pass 1, antes del refinamiento. Puedes interrumpir también durante la nueva etapa.

En modo `full` o con Pass 2 desactivado, estos controles no se aplican. Refinar añade tiempo de muestreo y todavía procesa todos los tokens del vídeo a la resolución de Pass 1: congelarlos no elimina su coste de memoria. No garantiza evitar OOM si esa resolución ya es demasiado alta, ni reproducir exactamente el audio del muestreo completo a alta resolución.

El ruido global y la acumulación se mantienen en CPU. El sampler recibe solo una ventana cada vez, aunque sigue necesitando memoria para el modelo y las referencias. El solapamiento se usa como contexto, pero cada posición final se asigna principalmente a la ventana cuyo centro está más cerca. Sólo se hace un fundido corto de dos posiciones latentes en el centro de cada unión; no se promedian predicciones independientes durante todo el solapamiento, evitando imágenes dobles y costuras bruscas. El nodo muestra `Chunk N/total` y permite interrumpir durante el muestreo o entre ventanas.

Este modo reduce el contexto temporal procesado simultáneamente; puede evitar OOM de Pass 2, pero no garantiza que cualquier resolución quepa ni que sea más rápido. Las transiciones pueden diferir del muestreo completo. No incluye tiles espaciales y no reduce la memoria del VAE al decodificar el vídeo final; para OOM de decodificación utiliza un decodificador VAE tiled.

## Preview y cancelación

Activa `enable_pass1_preview` y conecta `video_vae`. Para escuchar el audio de Pass 1, activa `preview_audio` y conecta `audio_vae`. `preview_fps` debe coincidir con los FPS de tu generación, normalmente 24.

El vídeo aparece dentro del nodo al terminar su decodificación, **antes del upscale**, y permanece visible durante Pass 2. El procesamiento continúa automáticamente; no espera aprobación. **Stop current execution** interrumpe el prompt que está ejecutando ComfyUI. También puedes utilizar la cancelación normal de ComfyUI. La interrupción se comprueba entre etapas y bloques del upscaler; una operación de GPU que ya comenzó tiene que devolver el control.

Con el preview desactivado no se decodifica ni se escribe un vídeo temporal nuevo y se oculta el reproductor anterior. Los previews se guardan en el directorio temporal de ComfyUI. El preview tiene coste de decodificación y puede provocar descarga/recarga del modelo de difusión para liberar VRAM.

Las pestañas **Pass 1 / Upscale / Pass 2 / Preview** muestran los ajustes relevantes. Toda la interfaz está implementada en JavaScript. Cambiar opciones del nodo puede volver a ejecutar Pass 1: no existe caché independiente por etapa.

## Upscaler 3D

El port está incluido; no necesitas mantener instalado el custom node original. Coloca sus pesos 3D en `ComfyUI/models/latent_upscale_models/`; admite subcarpetas, `.safetensors`, `.pth` de tensores y checkpoints con prefijo `upscaler.`. Los pesos del upscaler 2D no son intercambiables.

- **Multiplicador:** 1–4, con valor inicial 2.
- **Dimensiones:** ancho y alto objetivo independientes, en píxeles.
- **Megapíxeles:** mantiene la proporción antes de alinear; utiliza 1024 × 1024 píxeles por megapíxel, igual que el original.
- **Alineación:** primero redondea al múltiplo solicitado y luego a la cuadrícula espacial del VAE de 16 píxeles. El valor inicial es 32.
- **Temporal chunking:** conserva el algoritmo original, con bloques de 32 posiciones del latente, solapamiento según el kernel temporal y mezcla ponderada. No cambia la duración.
- **Dispositivo / precisión:** CUDA, ROCm o CPU; FP16, FP32 o BF16. La disponibilidad depende del hardware y de PyTorch.
- **Force unload:** descarga los pesos a CPU tras el upscale. Desactivarlo deja la residencia bajo la gestión de ComfyUI, que puede descargarlos para Pass 2. Se conserva como máximo un checkpoint por instancia de nodo; cambiar archivo, dispositivo o precisión invalida esa caché.

El audio se conserva durante el upscale. También se conservan los metadatos y se adapta la máscara espacial del vídeo. El modelo no añade interpolación temporal ni muestreo por tiles.

La arquitectura, estadísticas de normalización y algoritmo de chunking proceden de `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`, archivo `nodes/minimax_h3_latent_upscaler_3d.py`. Se adaptaron la carga segura de pesos, la gestión de memoria, los metadatos y la cancelación.

## Comprobación

Con el Python del entorno de ComfyUI:

```sh
python -m unittest discover -s tests -v
node tests/ui_test.mjs /ruta/al/ejecutable/chromium
```

Las pruebas cubren las cuatro combinaciones de etapas, selección de modelo, preview/cancelación, codificación de vídeo con audio, máscaras, carga y caché. Para el sampler temporal comprueban cobertura, fase temporal, ruido inicial compartido, guías, conservación del audio y cancelación. Incluyen una prueba del LCM nativo con un modelo H3 reducido y pesos sintéticos en CPU. Si está instalado el upscaler original, también comparan sus resultados con el port usando pesos pequeños, con y sin chunking. La prueba de navegador usa un entorno de interfaz simulado; estas comprobaciones no sustituyen una generación completa con el checkpoint real en GPU.
