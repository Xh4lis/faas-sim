resnet50_inference_cpu_manifest = "faas-workloads/resnet-inference-cpu"
resnet50_inference_gpu_manifest = "faas-workloads/resnet-inference-gpu"
resnet50_inference_function = "resnet50-inference"

resnet50_training_gpu_manifest = "faas-workloads/resnet-training-gpu"
resnet50_training_cpu_manifest = "faas-workloads/resnet-training-cpu"
resnet50_training_function = "resnet50-training"

speech_inference_tflite_manifest = "faas-workloads/speech-inference-tflite"
speech_inference_gpu_manifest = "faas-workloads/speech-inference-gpu"
speech_inference_function = "speech-inference"

mobilenet_inference_tflite_manifest = "faas-workloads/mobilenet-inference-tflite"
mobilenet_inference_tpu_manifest = "faas-workloads/mobilenet-inference-tpu"
mobilenet_inference_function = "mobilenet-inference"

resnet50_preprocessing_function = "resnet50-preprocessing"
resnet50_preprocessing_manifest = "faas-workloads/resnet-preprocessing"

pi_manifest = "faas-workloads/python-pi"
pi_function = "python-pi"

fio_manifest = "faas-workloads/fio"
fio_function = "fio"

# Add these new function definitions after the existing ones:
tf_gpu_manifest = "faas-workloads/tf-gpu"
tf_gpu_function = "tf-gpu"

# NEW SMART CITY FUNCTIONS - Add these lines
video_analytics_manifest = "faas-workloads/video-analytics"
video_analytics_function = "video-analytics"

iot_data_processor_manifest = "faas-workloads/iot-data-processor"
iot_data_processor_function = "iot-data-processor"

alert_service_manifest = "faas-workloads/alert-service"
alert_service_function = "alert-service"

data_aggregator_manifest = "faas-workloads/data-aggregator"
data_aggregator_function = "data-aggregator"

# REDUCE THE BIG EXISTING IMAGES (10x smaller):
all_ai_images = [
    # RESNET INFERENCE - Reduce from 2000M to 200M:
    (resnet50_inference_cpu_manifest, "200M", "x86"),      # Reduced from 2000M
    (resnet50_inference_cpu_manifest, "200M", "amd64"),    # Reduced from 2000M
    (resnet50_inference_cpu_manifest, "70M", "arm32v7"),   # Reduced from 700M
    (resnet50_inference_cpu_manifest, "70M", "arm32"),     # Reduced from 700M
    (resnet50_inference_cpu_manifest, "70M", "arm"),       # Reduced from 700M
    (resnet50_inference_cpu_manifest, "84M", "aarch64"),   # Reduced from 840M
    (resnet50_inference_cpu_manifest, "84M", "arm64"),     # Reduced from 840M
    (resnet50_training_cpu_manifest, "70M", "arm32v7"),    # Missing ARM32
    (resnet50_training_cpu_manifest, "70M", "arm32"),      # Missing ARM32
    (resnet50_training_cpu_manifest, "70M", "arm"),        # Missing ARM32
    (resnet50_training_cpu_manifest, "84M", "aarch64"),    # Missing ARM64
    (resnet50_training_cpu_manifest, "84M", "arm64"),      # Missing ARM64

    # RESNET GPU - Reduce from 2000M/1000M:
    (resnet50_inference_gpu_manifest, "200M", "x86"),      # Reduced from 2000M
    (resnet50_inference_gpu_manifest, "200M", "amd64"),    # Reduced from 2000M
    (resnet50_inference_gpu_manifest, "100M", "aarch64"),  # Reduced from 1000M
    (resnet50_inference_gpu_manifest, "100M", "arm64"),    # Reduced from 1000M
    
    # RESNET TRAINING - Reduce from 2000M/1000M:
    (resnet50_training_gpu_manifest, "200M", "x86"),       # Reduced from 2000M
    (resnet50_training_gpu_manifest, "200M", "amd64"),     # Reduced from 2000M
    (resnet50_training_gpu_manifest, "100M", "aarch64"),   # Reduced from 1000M
    (resnet50_training_gpu_manifest, "100M", "arm64"),     # Reduced from 1000M
    (resnet50_training_cpu_manifest, "200M", "amd64"),     # Reduced from 2000M
    (resnet50_training_cpu_manifest, "200M", "x86"),       # Reduced from 2000M
    
    # SPEECH INFERENCE - Keep smaller ones as-is, reduce big ones:
    (speech_inference_tflite_manifest, "108M", "amd64"),   # Keep
    (speech_inference_tflite_manifest, "108M", "x86"),     # Keep
    (speech_inference_tflite_manifest, "33M", "arm32v7"),  # Reduced from 328M
    (speech_inference_tflite_manifest, "33M", "arm32"),    # Reduced from 328M
    (speech_inference_tflite_manifest, "33M", "arm"),      # Reduced from 328M
    (speech_inference_tflite_manifest, "40M", "arm64"),    # Reduced from 400M
    (speech_inference_tflite_manifest, "40M", "aarch64"),  # Reduced from 400M
    (speech_inference_gpu_manifest, "160M", "amd64"),      # Reduced from 1600M
    (speech_inference_gpu_manifest, "160M", "x86"),        # Reduced from 1600M
    (speech_inference_gpu_manifest, "130M", "arm64"),      # Reduced from 1300M
    (speech_inference_gpu_manifest, "130M", "aarch64"),    # Reduced from 1300M
    
    # MOBILENET - Keep these, they're reasonable:
    (mobilenet_inference_tflite_manifest, "180M", "amd64"),
    (mobilenet_inference_tflite_manifest, "180M", "x86"),
    (mobilenet_inference_tflite_manifest, "160M", "arm32v7"),
    (mobilenet_inference_tflite_manifest, "160M", "arm32"),
    (mobilenet_inference_tflite_manifest, "160M", "arm"),
    (mobilenet_inference_tflite_manifest, "173M", "arm64"),
    (mobilenet_inference_tflite_manifest, "173M", "aarch64"),
    (mobilenet_inference_tpu_manifest, "173M", "arm64"),
    (mobilenet_inference_tpu_manifest, "173M", "aarch64"),
    
    # KEEP SMALL FUNCTIONS AS-IS:
    (pi_manifest, "88M", "amd64"),
    (pi_manifest, "88M", "x86"),
    (pi_manifest, "55M", "arm32v7"),
    (pi_manifest, "55M", "arm32"),
    (pi_manifest, "55M", "arm"),
    (pi_manifest, "62M", "arm64"),
    (pi_manifest, "62M", "aarch64"),
    (fio_manifest, "24M", "amd64"),
    (fio_manifest, "24M", "x86"),
    (fio_manifest, "20M", "arm32v7"),
    (fio_manifest, "20M", "arm32"),
    (fio_manifest, "20M", "arm"),
    (fio_manifest, "23M", "arm64"),
    (fio_manifest, "23M", "aarch64"),
    
    # TENSORFLOW - MASSIVE REDUCTION from 4100M/2240M:
    (tf_gpu_manifest, "410M", "amd64"),                     # Reduced from 4100M
    (tf_gpu_manifest, "410M", "x86"),                       # Reduced from 4100M
    (tf_gpu_manifest, "224M", "arm64"),                     # Reduced from 2240M
    (tf_gpu_manifest, "224M", "aarch64"),                   # Reduced from 2240M
    
    # RESNET PREPROCESSING - MASSIVE REDUCTION from 4100M/1370M/1910M:
    (resnet50_preprocessing_manifest, "410M", "x86"),       # Reduced from 4100M
    (resnet50_preprocessing_manifest, "410M", "amd64"),     # Reduced from 4100M
    (resnet50_preprocessing_manifest, "137M", "arm32v7"),   # Reduced from 1370M
    (resnet50_preprocessing_manifest, "137M", "arm32"),     # Reduced from 1370M
    (resnet50_preprocessing_manifest, "137M", "arm"),       # Reduced from 1370M
    (resnet50_preprocessing_manifest, "191M", "arm64"),     # Reduced from 1910M
    (resnet50_preprocessing_manifest, "191M", "aarch64"),   # Reduced from 1910M
    
#     # KEEP YOUR SMART CITY FUNCTIONS AS-IS (they're already reasonable):
#     (video_analytics_manifest, "80M", "x86"),
#     (video_analytics_manifest, "80M", "amd64"),
#     (video_analytics_manifest, "60M", "arm32v7"),
#     (video_analytics_manifest, "60M", "arm32"),
#     (video_analytics_manifest, "60M", "arm"),
#     (video_analytics_manifest, "70M", "arm64"),
#     (video_analytics_manifest, "70M", "aarch64"),
    
#     (iot_data_processor_manifest, "80M", "x86"),
#     (iot_data_processor_manifest, "80M", "amd64"),
#     (iot_data_processor_manifest, "70M", "arm32v7"),
#     (iot_data_processor_manifest, "70M", "arm32"),
#     (iot_data_processor_manifest, "70M", "arm"),
#     (iot_data_processor_manifest, "75M", "arm64"),
#     (iot_data_processor_manifest, "75M", "aarch64"),
    
#     (alert_service_manifest, "30M", "x86"),
#     (alert_service_manifest, "30M", "amd64"),
#     (alert_service_manifest, "25M", "arm32v7"),
#     (alert_service_manifest, "25M", "arm32"),
#     (alert_service_manifest, "25M", "arm"),
#     (alert_service_manifest, "28M", "arm64"),
#     (alert_service_manifest, "28M", "aarch64"),
    
#     (data_aggregator_manifest, "20M", "x86"),
#     (data_aggregator_manifest, "20M", "amd64"),
#     (data_aggregator_manifest, "18M", "arm32v7"),
#     (data_aggregator_manifest, "18M", "arm32"),
#     (data_aggregator_manifest, "18M", "arm"),
#     (data_aggregator_manifest, "19M", "arm64"),
#     (data_aggregator_manifest, "19M", "aarch64"),
# ]

# SMART CITY FUNCTIONS - REDUCE TO 10MB EACH:
    (video_analytics_manifest, "10M", "x86"),         # Reduced from 80M
    (video_analytics_manifest, "10M", "amd64"),       # Reduced from 80M
    (video_analytics_manifest, "8M", "arm32v7"),      # Reduced from 60M
    (video_analytics_manifest, "8M", "arm32"),        # Reduced from 60M
    (video_analytics_manifest, "8M", "arm"),          # Reduced from 60M
    (video_analytics_manifest, "9M", "arm64"),        # Reduced from 70M
    (video_analytics_manifest, "9M", "aarch64"),      # Reduced from 70M
    
    (iot_data_processor_manifest, "5M", "x86"),       # Reduced from 80M
    (iot_data_processor_manifest, "5M", "amd64"),     # Reduced from 80M
    (iot_data_processor_manifest, "4M", "arm32v7"),   # Reduced from 70M
    (iot_data_processor_manifest, "4M", "arm32"),     # Reduced from 70M
    (iot_data_processor_manifest, "4M", "arm"),       # Reduced from 70M
    (iot_data_processor_manifest, "4M", "arm64"),     # Reduced from 75M
    (iot_data_processor_manifest, "4M", "aarch64"),   # Reduced from 75M
    
    (alert_service_manifest, "3M", "x86"),            # Reduced from 30M
    (alert_service_manifest, "3M", "amd64"),          # Reduced from 30M
    (alert_service_manifest, "2M", "arm32v7"),        # Reduced from 25M
    (alert_service_manifest, "2M", "arm32"),          # Reduced from 25M
    (alert_service_manifest, "2M", "arm"),            # Reduced from 25M
    (alert_service_manifest, "3M", "arm64"),          # Reduced from 28M
    (alert_service_manifest, "3M", "aarch64"),        # Reduced from 28M
    
    (data_aggregator_manifest, "5M", "x86"),          # Reduced from 20M
    (data_aggregator_manifest, "5M", "amd64"),        # Reduced from 20M
    (data_aggregator_manifest, "4M", "arm32v7"),      # Reduced from 18M
    (data_aggregator_manifest, "4M", "arm32"),        # Reduced from 18M
    (data_aggregator_manifest, "4M", "arm"),          # Reduced from 18M
    (data_aggregator_manifest, "4M", "arm64"),        # Reduced from 19M
    (data_aggregator_manifest, "4M", "aarch64"),      # Reduced from 19M
]
# all_ai_images = [
    # (resnet50_inference_cpu_manifest, "2000M", "x86"),
    # (resnet50_inference_cpu_manifest, "2000M", "amd64"),
    # (resnet50_inference_cpu_manifest, "700M", "arm32v7"),
    # (resnet50_inference_cpu_manifest, "700M", "arm32"),
    # (resnet50_inference_cpu_manifest, "700M", "arm"),
    # (resnet50_inference_cpu_manifest, "840M", "aarch64"),
    # (resnet50_inference_cpu_manifest, "840M", "arm64"),
    # (resnet50_inference_gpu_manifest, "2000M", "x86"),
    # (resnet50_inference_gpu_manifest, "2000M", "amd64"),
    # (resnet50_inference_gpu_manifest, "1000M", "aarch64"),
    # (resnet50_inference_gpu_manifest, "1000M", "arm64"),
    # (resnet50_training_gpu_manifest, "2000M", "x86"),
    # (resnet50_training_gpu_manifest, "2000M", "amd64"),
    # (resnet50_training_gpu_manifest, "1000M", "aarch64"),
    # (resnet50_training_gpu_manifest, "1000M", "arm64"),
    # (resnet50_training_cpu_manifest, "2000M", "amd64"),
    # (resnet50_training_cpu_manifest, "2000M", "x86"),
    # (speech_inference_tflite_manifest, "108M", "amd64"),
    # (speech_inference_tflite_manifest, "108M", "x86"),
    # (speech_inference_tflite_manifest, "328M", "arm32v7"),
    # (speech_inference_tflite_manifest, "328M", "arm32"),
    # (speech_inference_tflite_manifest, "328M", "arm"),
    # (speech_inference_tflite_manifest, "400M", "arm64"),
    # (speech_inference_tflite_manifest, "400M", "aarch64"),
    # (speech_inference_gpu_manifest, "1600M", "amd64"),
    # (speech_inference_gpu_manifest, "1600M", "x86"),
    # (speech_inference_gpu_manifest, "1300M", "arm64"),
    # (speech_inference_gpu_manifest, "1300M", "aarch64"),
    # (mobilenet_inference_tflite_manifest, "180M", "amd64"),
    # (mobilenet_inference_tflite_manifest, "180M", "x86"),
    # (mobilenet_inference_tflite_manifest, "160M", "arm32v7"),
    # (mobilenet_inference_tflite_manifest, "160M", "arm32"),
    # (mobilenet_inference_tflite_manifest, "160M", "arm"),
    # (mobilenet_inference_tflite_manifest, "173M", "arm64"),
    # (mobilenet_inference_tflite_manifest, "173M", "aarch64"),
    # (mobilenet_inference_tpu_manifest, "173M", "arm64"),
    # (mobilenet_inference_tpu_manifest, "173M", "aarch64"),
    # (pi_manifest, "88M", "amd64"),
    # (pi_manifest, "88M", "x86"),
    # (pi_manifest, "55M", "arm32v7"),
    # (pi_manifest, "55M", "arm32"),
    # (pi_manifest, "55M", "arm"),
    # (pi_manifest, "62M", "arm64"),
    # (pi_manifest, "62M", "aarch64"),
    # (fio_manifest, "24M", "amd64"),
    # (fio_manifest, "24M", "x86"),
    # (fio_manifest, "20M", "arm32v7"),
    # (fio_manifest, "20M", "arm32"),
    # (fio_manifest, "20M", "arm"),
    # (fio_manifest, "23M", "arm64"),
    # (fio_manifest, "23M", "aarch64"),
    # (tf_gpu_manifest, "4100M", "amd64"),
    # (tf_gpu_manifest, "4100M", "x86"),
    # (tf_gpu_manifest, "2240M", "arm64"),
    # (tf_gpu_manifest, "2240M", "aarch64"),
    # (resnet50_preprocessing_manifest, "4100M", "x86"),
    # (resnet50_preprocessing_manifest, "4100M", "amd64"),
    # (resnet50_preprocessing_manifest, "1370M", "arm32v7"),
    # (resnet50_preprocessing_manifest, "1370M", "arm32"),
    # (resnet50_preprocessing_manifest, "1370M", "arm"),
    # (resnet50_preprocessing_manifest, "1910M", "arm64"),
    # (resnet50_preprocessing_manifest, "1910M", "aarch64"),
    # # VIDEO ANALYTICS - Larger images due to video processing libraries
    # (video_analytics_manifest, "800M", "x86"),
    # (video_analytics_manifest, "800M", "amd64"),
    # (video_analytics_manifest, "600M", "arm32v7"),
    # (video_analytics_manifest, "600M", "arm32"),
    # (video_analytics_manifest, "600M", "arm"),
    # (video_analytics_manifest, "700M", "arm64"),
    # (video_analytics_manifest, "700M", "aarch64"),
    
    # # IOT DATA PROCESSOR - Lightweight data processing
    # (iot_data_processor_manifest, "80M", "x86"),
    # (iot_data_processor_manifest, "80M", "amd64"),
    # (iot_data_processor_manifest, "70M", "arm32v7"),
    # (iot_data_processor_manifest, "70M", "arm32"),
    # (iot_data_processor_manifest, "70M", "arm"),
    # (iot_data_processor_manifest, "75M", "arm64"),
    # (iot_data_processor_manifest, "75M", "aarch64"),
    
    # # ALERT SERVICE - Very lightweight, minimal dependencies
    # (alert_service_manifest, "30M", "x86"),
    # (alert_service_manifest, "30M", "amd64"),
    # (alert_service_manifest, "25M", "arm32v7"),
    # (alert_service_manifest, "25M", "arm32"),
    # (alert_service_manifest, "25M", "arm"),
    # (alert_service_manifest, "28M", "arm64"),
    # (alert_service_manifest, "28M", "aarch64"),
    
    # # DATA AGGREGATOR - Analytics libraries, medium size
    # (data_aggregator_manifest, "200M", "x86"),
    # (data_aggregator_manifest, "200M", "amd64"),
    # (data_aggregator_manifest, "180M", "arm32v7"),
    # (data_aggregator_manifest, "180M", "arm32"),
    # (data_aggregator_manifest, "180M", "arm"),
    # (data_aggregator_manifest, "190M", "arm64"),
    # (data_aggregator_manifest, "190M", "aarch64"),
# ]
