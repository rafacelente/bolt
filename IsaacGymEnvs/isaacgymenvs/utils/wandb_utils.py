from rl_games.common.algo_observer import AlgoObserver

from isaacgymenvs.utils.utils import retry
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import wandb
#import torch
import os

class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg, local_run_dir):
        super().__init__()
        self.cfg = cfg
        self.local_run_dir = local_run_dir
        self.algo = None
        self.writer = None

        self.ep_infos = []
        self.direct_info = {}

        self.episode_cumulative = dict()
        self.episode_cumulative_avg = dict()
        self.new_finished_episodes = False

    def log_model_to_wandb(self, name):
        if self.log_model:
            print('Logging model to wandb...')
            best_model_path = os.path.join(self.local_run_dir, f'nn/{name}.pth')
            best_model = wandb.Artifact(f"model_{self.run_id}", type='model')
            best_model.add_file(best_model_path)
            wandb.run.log_artifact(best_model)

            # # Logging video
            # run_name = os.path.basename(self.local_run_dir)
            # video_path = os.path.join("/home/odri/git/pie2023/IsaacGymEnvs/isaacgymenvs/videos", os.path.join(run_name, "rl-video-step-2000.mp4"))
            # run_video = wandb.Artifact(f"video_{self.run_id}", type='video')
            # run_video.add_file(video_path)
            # wandb.run.log_artifact(run_video)

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        self.log_model = self.cfg.get('wandb_log_model', False)

        wandb_unique_id = f"{experiment_name}"
        self.run_id = None
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method='fork'),
            )
            
            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print('wandb running directory........', wandb.run.dir)

        print('Initializing WandB...')
        try:
            init_wandb()
            self.run_id = wandb.run.id
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    # def after_init(self, algo):
    #     self.algo = algo

    # def process_infos(self, infos, done_indices):
    #     assert isinstance(infos, dict), 'RLGPUAlgoObserver expects dict info'
    #     if not isinstance(infos, dict):
    #         return

    #     if 'episode' in infos:
    #         self.ep_infos.append(infos['episode'])

    #     if 'episode_cumulative' in infos:
    #         for key, value in infos['episode_cumulative'].items():
    #             if key not in self.episode_cumulative:
    #                 self.episode_cumulative[key] = torch.zeros_like(value)
    #             self.episode_cumulative[key] += value

    #         for done_idx in done_indices:
    #             self.new_finished_episodes = True
    #             done_idx = done_idx.item()

    #             for key, value in infos['episode_cumulative'].items():
    #                 if key not in self.episode_cumulative_avg:
    #                     self.episode_cumulative_avg[key] = deque([], maxlen=self.algo.games_to_track)

    #                 self.episode_cumulative_avg[key].append(self.episode_cumulative[key][done_idx].item())
    #                 self.episode_cumulative[key][done_idx] = 0

    #     # turn nested infos into summary keys (i.e. infos['scalars']['lr'] -> infos['scalars/lr']
    #     if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
    #         infos_flat = flatten_dict(infos, prefix='', separator='/')
    #         self.direct_info = {}
    #         for k, v in infos_flat.items():
    #             # only log scalars
    #             if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
    #                 self.direct_info[k] = v

    # def after_print_stats(self, frame, epoch_num, total_time):
    #     if self.ep_infos:
    #         for key in self.ep_infos[0]:
    #             infotensor = torch.tensor([], device=self.algo.device)
    #             for ep_info in self.ep_infos:
    #                 # handle scalar and zero dimensional tensor infos
    #                 if not isinstance(ep_info[key], torch.Tensor):
    #                     ep_info[key] = torch.Tensor([ep_info[key]])
    #                 if len(ep_info[key].shape) == 0:
    #                     ep_info[key] = ep_info[key].unsqueeze(0)
    #                 infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
    #             value = torch.mean(infotensor)
    #             self.writer.add_scalar('Episode/' + key, value, epoch_num)
    #         self.ep_infos.clear()
        
    #     # log these if and only if we have new finished episodes
    #     if self.new_finished_episodes:
    #         for key in self.episode_cumulative_avg:
    #             self.writer.add_scalar(f'episode_cumulative/{key}', np.mean(self.episode_cumulative_avg[key]), frame)
    #             self.writer.add_scalar(f'episode_cumulative_min/{key}_min', np.min(self.episode_cumulative_avg[key]), frame)
    #             self.writer.add_scalar(f'episode_cumulative_max/{key}_max', np.max(self.episode_cumulative_avg[key]), frame)
    #         self.new_finished_episodes = False

    #     for k, v in self.direct_info.items():
    #         self.writer.add_scalar(f'{k}/frame', v, frame)
    #         self.writer.add_scalar(f'{k}/iter', v, epoch_num)
    #         self.writer.add_scalar(f'{k}/time', v, total_time)

    