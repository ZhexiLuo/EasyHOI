import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def should_run_icp(current_iter: int, icp_iters_range) -> bool:
    """
    判断当前迭代轮次是否应该运行ICP配准。
    
    支持两种配置格式：
    1. 单区间格式：[start, end] - 在[start, end)区间内运行ICP
    2. 多区间格式：[[start1, end1], [start2, end2], ...] - 在多个区间内运行ICP
    
    Args:
        current_iter (int): 当前迭代轮次
        icp_iters_range: ICP配准的轮次配置
            - 单区间：[start, end] 
            - 多区间：[[start1, end1], [start2, end2], ...]
    
    Returns:
        bool: 如果当前轮次应该运行ICP则返回True，否则返回False
    
    Examples:
        >>> # 单区间配置
        >>> should_run_icp(1, [0, 5])  # True (1 在 [0,5) 内)
        >>> should_run_icp(5, [0, 5])  # False (5 不在 [0,5) 内)
        
        >>> # 多区间配置  
        >>> should_run_icp(1, [[0, 2], [5, 8]])   # True (1 在 [0,2) 内)
        >>> should_run_icp(3, [[0, 2], [5, 8]])   # False (3 不在任何区间内)
        >>> should_run_icp(6, [[0, 2], [5, 8]])   # True (6 在 [5,8) 内)
    """
    if not icp_iters_range:
        return False
    
    # 检查是否为多区间格式
    if isinstance(icp_iters_range[0], (list, tuple)):
        # 多区间格式 [[start1, end1], [start2, end2], ...]
        for start, end in icp_iters_range:
            if start <= current_iter < end:
                return True
        return False
    else:
        # 单区间格式 [start, end] (向后兼容)
        start, end = icp_iters_range
        return start <= current_iter < end
