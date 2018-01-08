from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.summary import summary
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver

import contrib_evaluation as evaluation

_USE_DEFAULT = 0

def evaluate_once(master,
                  final_layer,
                  checkpoint_path,
                  logdir,
                  num_evals=1,
                  initial_op=None,
                  initial_op_feed_dict=None,
                  eval_op=None,
                  eval_op_feed_dict=None,
                  final_op=None,
                  final_op_feed_dict=None,
                  summary_op=_USE_DEFAULT,
                  summary_op_feed_dict=None,
                  variables_to_restore=None,
                  session_config=None,
                  hooks=None):
  """Evaluates the model at the given checkpoint path.
  Args:
    master: The BNS address of the TensorFlow master.
    checkpoint_path: The path to a checkpoint to use for evaluation.
    logdir: The directory where the TensorFlow summaries are written to.
    num_evals: The number of times to run `eval_op`.
    initial_op: An operation run at the beginning of evaluation.
    initial_op_feed_dict: A feed dictionary to use when executing `initial_op`.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: The summary_op to evaluate after running TF-Slims metric ops. By
      default the summary_op is set to tf.summary.merge_all().
    summary_op_feed_dict: An optional feed dictionary to use when running the
      `summary_op`.
    variables_to_restore: A list of TensorFlow variables to restore during
      evaluation. If the argument is left as `None` then
      slim.variables.GetVariablesToRestore() is used.
    session_config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    hooks: A list of additional `SessionRunHook` objects to pass during the
      evaluation.
  Returns:
    The value of `final_op` or `None` if `final_op` is `None`.
  """
  if summary_op == _USE_DEFAULT:
    summary_op = summary.merge_all()

  all_hooks = [evaluation.StopAfterNEvalsHook(num_evals),]

  if summary_op is not None:
    all_hooks.append(evaluation.SummaryAtEndHook(
        log_dir=logdir, summary_op=summary_op, feed_dict=summary_op_feed_dict))
  if hooks is not None:
    all_hooks.extend(hooks)

  saver = None
  if variables_to_restore is not None:
    saver = tf_saver.Saver(variables_to_restore)

  return evaluation.evaluate_once(
      checkpoint_path,
      master=master,
      final_layer=final_layer,
      scaffold=monitored_session.Scaffold(
          init_op=initial_op, init_feed_dict=initial_op_feed_dict, saver=saver),
      eval_ops=eval_op,
      feed_dict=eval_op_feed_dict,
      final_ops=final_op,
      final_ops_feed_dict=final_op_feed_dict,
      hooks=all_hooks,
      config=session_config)
