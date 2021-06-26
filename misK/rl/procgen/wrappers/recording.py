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

ACTIONS = [u'↙', u'←', u'↖', u'↓', u'⌀',
           u'↑', u'↘', u'→', u'↗', 'D',
           'A', 'W', 'S', 'Q', 'E']
LENGTH = 50
_f = np.pi / 180
ARROWS = [(135 * _f, LENGTH), (180 * _f, LENGTH), (225 * _f, LENGTH), (+90 * _f, LENGTH), (0, 0),
          (270 * _f, LENGTH), (+45 * _f, LENGTH), (+0. * _f, LENGTH), (315 * _f, LENGTH), (0, 0),
          (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]


class Recorder(VecEnvWrapper):
    def __init__(self, env, directory, name_prefix="gif", mode="episode", trigger=1,
                 frame_rate=30, needs_render=0, desired_output_size=(480, 480),
                 font="font.ttf", font_size=1, log=print):
        """
            Constructs a Recorder instance, to allow the recording of an agent in the environment.

            Args
            ----
            env : ToBaselinesVecEnv or child
                the vectorized environment that needs some recording.
            directory : str
                the name of the root directory used for the recordings.
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

        # to store the frames.
        self.current_frame = None
        self.frame_buffer = []  # contains all the frames that need to be recorded.
        self.meta_buffer = []  # contains the meta data about the episodes, namely episode lengths and endings.

        # the save mode.
        self.mode = mode  # the saving mode.
        self.trigger = trigger  # the trigger value in trigger mode.
        self.start_ep = 1  # the starting episode to correctly label the ouput video file.
        self.step_count = 0  # the current step inside the current episode.

        # the save parameters.
        self.desired_output_size = (int(desired_output_size[0]), int(desired_output_size[1]))
        self.frame_rate = frame_rate  # the frame rate for the ouput video.
        self.directory = directory  # the place to store the video in.
        self.name_prefix = name_prefix  # the name prefix of the video file.

        # text parameters.
        self.text_color = (255, 255, 255)
        self.font = ImageFont.truetype(font, size=int(font_size))
        ""

        self.current_action = 0  # the current action taken by the agent.

        # final monitoring.
        self.saved_videos = []

        # triggers the live rendering of the environment.
        self.needs_render = needs_render

        # directory auto creation.
        autocreation = False
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            autocreation = True
        self.print(f"(recordings will be saved in {self.directory} "
                   f"(name: {name_prefix}, auto: {int(autocreation)}))", end=' ')

        self.dist_buffer = []
        self.action_buffer = []

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
        self.current_frame = obs[0]
        self.frame_buffer.append(self.current_frame)
        return obs

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
        self.action_buffer.append(actions[0])
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
        self.step_count += 1  # always increment the step count before pushing the frame.

        if sum(dones) > 0:
            # the episode is done.
            # add the info about the end of episode.
            self.meta_buffer += [(self.step_count, "reward" if sum(rewards) > 0 else "dead")]
            self.step_count = 0

            if self.mode == "episode" or (self.mode == "trigger" and len(self.meta_buffer) % self.trigger == 0):
                # needs to save a video
                if len(self.frame_buffer) > 2:
                    # only if there is something to save.
                    self.save()

        else:

            # store the new frame into the buffer.
            self.current_frame = obs[0]
            self.frame_buffer.append(self.current_frame)

        if self.needs_render:
            self._render(time=self.needs_render)

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
            arrow_a, arrow_l = ARROWS[self.action_buffer[i]]
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
            draw.rectangle(tuple(map(sum, zip(rect, offset))), fill=tuple(map(int, [gray] * 3)), outline=box, width=3)

        # file stamps.
        now = datetime.now()
        name = f"{self.start_ep}-{len(self.meta_buffer)}_"
        name += now.strftime("%d-%m-%Y_%H-%M-%S") + '_' + str(time.time_ns())
        filename = os.path.join(self.directory, name + ".gif")

        # saving the video.
        self.print(f"saving video in {filename}...", end=' ', flush=True)
        print("saving video")
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
        if len(self.frame_buffer) > 1:
            self.save()

        # show the saved videos and clears the list to print it only once when Recorder.close() is called many times.
        if len(self.saved_videos) > 0:
            self.print("open saved videos with:")
            for video in self.saved_videos:
                self.print(f"eog {video}")
            self.saved_videos = []


def get_text_dimensions(text_string, font):
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()

    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent

    return text_width, text_height
