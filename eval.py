# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Evaluation executable for detection models.

This executable is used to evaluate DetectionModels. There are two ways of
configuring the eval job.

1) A single pipeline_pb2.TrainEvalPipelineConfig file maybe specified instead.
In this mode, the --eval_training_data flag may be given to force the pipeline
to evaluate on training data instead.

Example usage:
python eval.py --logtostderr --train_dir=/path/to/train/dir

"""
import functools
import os
import tensorflow as tf

from object_detection import evaluator
from object_detection.builders import dataset_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('train_dir', '', 'Base folder of the experiment containing the train_logs folder')
flags.DEFINE_boolean('run_once', False, 'Option to only run a single pass of '
                     'evaluation. Overrides the `max_evals` parameter in the '
                     'provided config.')
FLAGS = flags.FLAGS


def main(unused_argv):
  assert FLAGS.train_dir, '`train_dir` is missing.'

  eval_dir = os.path.join(FLAGS.train_dir,'eval_logs')
  ckpt_dir = os.path.join(FLAGS.train_dir,'train_logs')

  tf.gfile.MakeDirs(eval_dir)

  pipeline_config_path = os.path.join(FLAGS.train_dir,'model.config')
  configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
  tf.gfile.Copy(pipeline_config_path, os.path.join(eval_dir, 'pipeline.config'), overwrite=True)


  model_config = configs['model']
  eval_config = configs['eval_config']
  input_config = configs['eval_input_config']

  model_fn = functools.partial(
    model_builder.build, 
    model_config=model_config,
    is_training=False)

  def get_next(config):
    return dataset_util.make_initializable_iterator(dataset_builder.build(config)).get_next()

  create_input_dict_fn = functools.partial(get_next, input_config)

  label_map = label_map_util.load_labelmap(input_config.label_map_path)
  max_num_classes = max([item.id for item in label_map.item])
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes)

  if FLAGS.run_once:
    eval_config.max_evals = 1

  best_eval_metrics = {
    'best_mAP': 0.0,
    'best_tot_loss': 999.9
  }

  evaluator.evaluate(create_input_dict_fn, model_fn, eval_config, categories, ckpt_dir, eval_dir, best_eval_metrics=best_eval_metrics)

if __name__ == '__main__':
  tf.app.run()
