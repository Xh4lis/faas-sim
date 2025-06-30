from ether.util import parse_size_string
from skippy.core.storage import DataItem

resnet_train_bucket = "bucket_resnet50_train"
resnet_pre_bucket = "bucket_resnet50_pre"
resnet_model_bucket = "bucket_resnet50_model"
speech_bucket = "bucket_speech"
mobilenet_bucket = "mobilenet_bucket"

# ADD NEW SMART CITY BUCKETS:
video_analytics_bucket = "bucket_video_analytics"
iot_data_bucket = "bucket_iot_data" 
alert_service_bucket = "bucket_alert_service"
data_aggregator_bucket = "bucket_data_aggregator"



resnet_train_bucket_item = DataItem(
    resnet_train_bucket, "raw_data", parse_size_string("58M")
)
resnet_pre_bucket_item = DataItem(
    resnet_pre_bucket, "raw_data", parse_size_string("14M")
)
resnet_model_bucket_item = DataItem(
    resnet_model_bucket, "model", parse_size_string("103M")
)
speech_model_tflite_bucket_item = DataItem(
    speech_bucket, "model_tflite", parse_size_string("48M")
)
speech_model_gpu_bucket_item = DataItem(
    speech_bucket, "model_gpu", parse_size_string("188M")
)
mobilenet_model_tflite_bucket_item = DataItem(
    mobilenet_bucket, "model_tflite", parse_size_string("4M")
)
mobilenet_model_tpu_bucket_item = DataItem(
    mobilenet_bucket, "model_tpu", parse_size_string("4M")
)

# ADD NEW DATA ITEMS:
video_analytics_model_item = DataItem(
    video_analytics_bucket, "video_model", parse_size_string("150M")
)
iot_config_item = DataItem(
    iot_data_bucket, "iot_config", parse_size_string("5M")
)
alert_templates_item = DataItem(
    alert_service_bucket, "alert_templates", parse_size_string("2M")
)
aggregator_data_item = DataItem(
    data_aggregator_bucket, "historical_data", parse_size_string("200M")
)


bucket_names = [
    resnet_model_bucket,
    resnet_train_bucket,
    resnet_pre_bucket,
    mobilenet_bucket,
    speech_bucket,
    # ADD NEW BUCKETS:
    video_analytics_bucket,
    iot_data_bucket,
    alert_service_bucket,
    data_aggregator_bucket,
]

data_items = [
    resnet_train_bucket_item,
    resnet_pre_bucket_item,
    resnet_model_bucket_item,
    speech_model_gpu_bucket_item,
    speech_model_tflite_bucket_item,
    mobilenet_model_tpu_bucket_item,
    mobilenet_model_tflite_bucket_item,
    # ADD NEW DATA ITEMS:
    video_analytics_model_item,
    iot_config_item,
    alert_templates_item,
    aggregator_data_item,
]
