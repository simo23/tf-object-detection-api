'''
Example Usage:
--------------

python export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory \
    --config_override " \
            model{ \
              faster_rcnn { \
                second_stage_post_processing { \
                  batch_non_max_suppression { \
                    score_threshold: 0.5 \
                  } \
                } \
              } \
            }"

'''

import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('conf', 
  None, 
  'Path to a config file.')
flags.DEFINE_string('ckpt', 
  None,
  'Path to checkpoint prefix file')
flags.DEFINE_string('output_dir', 
  None, 
  'Path to write outputs.')
flags.DEFINE_string('score_th', 
  0.25, 
  'Score threshold to use in the post_processing')
flags.DEFINE_string('iou_th', 
  0.6, 
  'IOU to use in the post_processing phase')
FLAGS = flags.FLAGS

def main(_):

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

  config_override = "model{ \
                        ssd { \
                          post_processing { \
                            batch_non_max_suppression { \
                              score_threshold: %s \
                              iou_threshold: %s \
                              max_detections_per_class: 20 \
                              max_total_detections: 20 \
                            } \
                          } \
                        } \
                    }" % (FLAGS.score_th,FLAGS.iou_th)
  
  with tf.gfile.GFile(FLAGS.conf, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  text_format.Merge(config_override, pipeline_config)

  exporter.export_inference_graph(
    'image_tensor', 
    pipeline_config,
    FLAGS.ckpt,
    FLAGS.output_dir,
    None)

if __name__ == '__main__':
  tf.app.run()
