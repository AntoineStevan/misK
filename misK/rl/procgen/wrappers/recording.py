import os
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import (ImageFont,
                 Image,
                 ImageDraw)
from matplotlib import pyplot as plt
from numpy import cos, sin
from scipy import ndimage
from tqdm.auto import trange

from misK.rl.procgen.wrappers.base import VecEnvWrapper

_actions = [u'↙', u'←', u'↖', u'↓', u'⌀',
            u'↑', u'↘', u'→', u'↗', 'D',
            'A', 'W', 'S', 'Q', 'E']
_length = 50
_f = np.pi / 180
_arrows = [(135 * _f, _length), (180 * _f, _length), (225 * _f, _length), (+90 * _f, _length), (0, 0),
           (270 * _f, _length), (+45 * _f, _length), (+0. * _f, _length), (315 * _f, _length), (0, 0),
           (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]


class Recorder(VecEnvWrapper):
    def __init__(self, env, directory, mode="episode", trigger=1,
                 frame_rate=30, needs_render=0, desired_output_size=(480, 480),
                 font="font.ttf", font_size=1, log=print):
        """
            Constructs a Recorder instance, to allow the recording of an agent in the environment.

            Args
            ----
            env : ToBaselinesVecEnv or child
                the vectorized environment that needs some recording.
            name_prefix : str, optional
                the name prefix for all the current recordings.
            mode : str, optional
                the recorde mode. If "episode", all episodes are saved in separate recordings. If "trigger", uses
                the trigger to record, i.e. in one file, there will be 'trigger' consecutive episodes. Otherwise,
                all the episodes, from constructing to closing the environment, will be in the same video.
            trigger : int, optional
                the number of episodes in each video, if mode is "trigger".
            frame_rate : int, optional
                the frame rate of the final videos.
            needs_render : int, optional
                tells if the wrapper needs to lively render the observations. it is indeed the time for each frame.
                If less than or equal to 0, no rendering.
            desired_output_size : (int, int), optional
                the output video size, in pixels.
            font : str, optional
                the path to the desired font.
            font_size : int, optional
                the font size for overlays.
            log : function, optional
                the log function to print strings. Defaults to built-in print.

            Returns
            -------
            self : Recorder
                the constructed Recorder object instance.
        """
        self.print = log

        self.print(f"-> {self.__class__.__name__}")
        super().__init__(venv=env)

        # # the save parameters.
        self.desired_output_size = (int(desired_output_size[0]), int(desired_output_size[1]))
        self.frame_rate = frame_rate  # the frame rate for the ouput video.
        self.directory = directory  # the place to store the video in.

        # triggers the live rendering of the environment.
        self.needs_render = needs_render

        now = datetime.now()
        name = now.strftime("%d-%m-%Y_%H-%M-%S") + '_' + str(time.time_ns())
        self.video_dir = os.path.join(self.directory, name)

        # directory auto creation.
        autocreation = False
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
            autocreation = True
        self.print(f"(recordings will be saved in {self.video_dir} (auto: {int(autocreation)}))", end=' ')

        self.dist_buffer = []
        self.action_buffer = [np.zeros(shape=(self.num_envs,))]

        self.frames = np.zeros(shape=(self.num_envs,))  # number of frames in each current environment in the vector.
        self.episodes = np.zeros(shape=(self.num_envs,))  # stashes the number of elapsed episodes in each environment.

    def push_dist(self, dist):
        self.dist_buffer.append(dist)

    def reset(self):
        """
            Wrapper for the reset method.
            Also adds frames to the buffer.

            Args
            ----

            Returns
            -------
            obs : (Tensor)
                gives the gym/procgen first observation in the environment.
        """
        obs = self.venv.reset()

        # store the first frame into the buffer.
        self.save_obs(obs, dones=[False] * self.num_envs, rewards=np.zeros(shape=(self.num_envs,)))
        return obs

    def save_obs(self, obs, dones, rewards):
        for env, frame in enumerate(obs):
            head = os.path.join(self.video_dir, f"{env:06d}_{self.episodes[env]:06.0f}")
            fname = head + f"_{self.frames[env]:06.0f}_{rewards[env]}_{self.action_buffer[-1][env]}_{dones[env]}.png"
            plt.imsave(fname, np.transpose(frame, (1, 2, 0)))

    def step_async(self, actions):
        """
            Wrapper for the step_async method.
            Also takes care of the actions taken for frame stamping.

            Args
            ----
            actions : list of ints, optional
                the actions taken in the vectorized environment.

            Returns
            -------
            None
        """
        self.action_buffer.append(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        """
            Wrapper for the step_wait method.
            Also adds frames to the buffer and saves the video if the mode needs it.

            Args
            ----

            Returns
            -------
            (obs, rewards, dones, infos) : (Tensor, float or int, bool, dict or None)
                gives the gym/procgen results for a step method.
        """

        obs, rewards, dones, infos = self.venv.step_wait()

        if self.needs_render:
            self._render(time=self.needs_render)

        self.save_obs(obs, dones, rewards)
        self.frames = (self.frames + 1) * (1 - dones)
        self.episodes += dones

        return obs, rewards, dones, infos

    def _render(self, time=50):
        """
            Renders a frame during 'time' milliseconds, using matplotlib.

            Args
            ----
            time : int, optional
                the duration of each frame, in milliseconds.

            Returns
            -------
            None
        """
        plt.imshow(np.transpose(self.current_frame, (1, 2, 0)))
        plt.pause(time / 1000)
        plt.cla()

    def save(self):
        """
            Saves the frame buffer inside a dedicated, unique, .gif file.
            Uses the name, the deco and the current time to create a unique stamp for each video.
            Finally clears the frame buffer for future videos.

            Args
            ----

            Returns
            -------
            None
        """
        # computing the overlay stamps.
        lengths, infos = list(zip(*self.meta_buffer))
        overlays = []
        t = trange(len(lengths), desc="computing overlay stamps")
        for i in t:
            overlays += [(i + 1, len(lengths), k + 1, lengths[i], infos[i]) for k in range(lengths[i])]

        # frame stamps.
        t = trange(len(self.frame_buffer), desc="image processing and stamping")
        player = tuple(map(lambda x: x // 2, self.desired_output_size))
        for i in t:
            self.frame_buffer[i] = np.transpose(self.frame_buffer[i], (1, 2, 0))
            self.frame_buffer[i] = (self.frame_buffer[i] * 255).astype(np.uint8)
            zoom = (self.desired_output_size[0] / self.frame_buffer[i].shape[0],
                    self.desired_output_size[1] / self.frame_buffer[i].shape[1], 1.)
            self.frame_buffer[i] = ndimage.zoom(self.frame_buffer[i], zoom=zoom, order=0)
            arrow_a, arrow_l = _arrows[self.action_buffer[i]]
            x, y = tuple(map(int, (cos(arrow_a) * arrow_l, sin(arrow_a) * arrow_l)))
            self.frame_buffer[i] = cv2.arrowedLine(self.frame_buffer[i], player, tuple(map(sum, zip(player, (x, y)))),
                                                   (255, 0, 0), 8)
            self.frame_buffer[i] = Image.fromarray(self.frame_buffer[i])

            draw = ImageDraw.Draw(self.frame_buffer[i])
            lines = [
                f".epi: {overlays[i][0]} / {overlays[i][1]}",
                f"step: {overlays[i][2]} / {overlays[i][3]}",
                f".end: {overlays[i][4]}",
            ]
            x_text, y_text = 0, 0
            for line in lines:
                w, h = get_text_dimensions(line, self.font)
                draw.rectangle((x_text, y_text, x_text + w, y_text + h), fill=(0, 0, 0))
                draw.text((x_text, y_text), line, self.text_color, font=self.font)
                y_text += h

            if self.dist_buffer[i] is not None:
                w, h = self.desired_output_size[0] // len(self.dist_buffer[i]), 50
                rect = (x_text, self.desired_output_size[1] - h, w - 1, self.desired_output_size[1] - 1)
                for j, comp in enumerate(self.dist_buffer[i]):
                    if j == self.action_buffer[i]:
                        continue
                    offset = (j * w, 0, j * w, 0)
                    gray = comp * 255

                    draw.rectangle(tuple(map(sum, zip(rect, offset))), fill=tuple(map(int, [gray] * 3)))
                offset = (self.action_buffer[i] * w, 0, self.action_buffer[i] * w, 0)
                gray = self.dist_buffer[i][self.action_buffer[i]] * 255
                box = (255, 0, 0)
                draw.rectangle(tuple(map(sum, zip(rect, offset))), fill=tuple(map(int, [gray] * 3)), outline=box,
                               width=3)

        # file stamps.
        now = datetime.now()
        name = f"{self.start_ep}-{len(self.meta_buffer)}_"
        name += now.strftime("%d-%m-%Y_%H-%M-%S") + '_' + str(time.time_ns())
        filename = os.path.join(self.directory, name + ".gif")

        # saving the video.
        self.print(f"saving video in {filename}...", end=' ', flush=True)
        self.frame_buffer[0].save(filename, save_all=True, append_images=self.frame_buffer[1:],
                                  optimize=True, duration=1000 // self.frame_rate, loop=0)
        self.saved_videos.append(filename)
        self.print("done")

        # clearing the buffers.
        self.clear_buffers()

    def clear_buffers(self):
        """
            Clears the buffers of the environment.

            Args
            ----

            Returns
            -------
            None
        """
        self.frame_buffer = []
        self.meta_buffer = []
        self.dist_buffer = []
        self.action_buffer = []
        self.start_ep = len(self.meta_buffer) + 1

    def close(self):
        """
            Wrappers for the close method.
            Saves the last frames in the buffer.

            Args
            ----

            Returns
            -------
            None
        """
        self.venv.close()
        # if len(self.frame_buffer) > 1:
        #     self.save()
        #
        # # show the saved videos and clears the list to print it only once when Recorder.close() is called many times.
        # if len(self.saved_videos) > 0:
        #     self.print("open saved videos with:")
        #     for video in self.saved_videos:
        #         self.print(f"eog {video}")
        #     self.saved_videos = []


def get_text_dimensions(text_string, font):
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()

    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent

    return text_width, text_height
