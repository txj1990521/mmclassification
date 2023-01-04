from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SonicAfterRunHook(Hook):

    def after_run(self, runner):
        pass
        # runner.after_run()  # 给训练结束后提供一个接口
