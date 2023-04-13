"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
from abc import ABC, abstractmethod

import cloudpickle
import psutil
import os
import numpy as np
from multiprocessing import Process, Pipe


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        pass
        #self.render(mode)
        # imgs = self.get_images()
        # bigimg = tile_images(imgs)
        # if mode == 'human':
        #     self.get_viewer().imshow(bigimg)
        #     return self.get_viewer().isopen
        # elif mode == 'rgb_array':
        #     return bigimg
        # else:
        #     raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(num_envs=venv.num_envs,
                         observation_space=observation_space or venv.observation_space,
                         action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            st, ob, reward, done, info = env.step(data)
            # if done:
            #     st, ob = env.reset()
            remote.send((st, ob, reward, done, info))
        elif cmd == 'reset':
            st, ob = env.reset()
            remote.send((st, ob))
        elif cmd == 'get_st_obs':
            st, ob = env.get_st_obs()
            remote.send((st, ob))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.state_space, env.observation_space, env.action_space))
        elif cmd == 'get_visit_counts':
            if hasattr(env, 'visit_counts'):
                remote.send(env.visit_counts)
            elif hasattr(env.unwrapped, 'visit_counts'):
                remote.send(env.unwrapped.visit_counts)
            else:
                raise NotImplementedError
        elif cmd == 'render':
            env.render()
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = False  # can't since vizdoom envs have their own daemonic subprocesses
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.state_space, observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def get_visit_counts(self):
        for remote in self.remotes:
            remote.send(('get_visit_counts', None))
        return sum(remote.recv() for remote in self.remotes)

    def step_async(self, actions, envs):
        for action, env_i in zip(actions, envs):
            self.remotes[env_i].send(('step', action))
        self.waiting = True

    def step_wait(self, envs):
        results = [self.remotes[i].recv() for i in envs]
        self.waiting = False
        state, obs, rews, dones, infos = zip(*results)
        # state and obs are tuples, so stack their components separately
        return (self._stack(state),
                self._stack(obs),
                np.array(rews), np.array(dones), infos)

    def _stack(self, items, *args):
        """
        Stack items received from multiple environments.
        items indexed as such: (n_envs, ..., numpy array with shape (*dims)) where '...' can be any number
        of arbitrary nested tuples/lists, to get a set of nested lists indexed as
        (..., numpy array w/ shape (n_envs, *dims))
        """
        if len(args) == 0:
            return self._stack(items, 0)
        sub_items = items
        for dim in args[::-1]:
            sub_items = sub_items[dim]
        if type(sub_items) in (tuple, list):
            return [self._stack(items, i, *args) for i in range(len(sub_items))]
        else:
            will_stack = []
            for i in range(len(items)):
                sub_items = items[i]
                for dim in args[:-1][::-1]:
                    sub_items = sub_items[dim]
                will_stack.append(sub_items)
            return np.stack(will_stack)

    def step(self, actions, env_mask=None):
        if env_mask is None:
            env_mask = np.ones(len(self.remotes))
        envs = np.where(env_mask)[0]
        self.step_async(actions, envs)
        return self.step_wait(envs)

    def reset(self, need_reset=None):
        if need_reset is None:
            need_reset = [True for _ in range(len(self.remotes))]
        for remote, nr in zip(self.remotes, need_reset):
            if nr:
                remote.send(('reset', None))
            else:
                remote.send(('get_st_obs', None))
        results = [remote.recv() for remote in self.remotes]
        state, obs = zip(*results)
        return (self._stack(state), self._stack(obs))

    def close(self, force=False):
        if self.closed:
            return
        if force:  # super ugly
            for p in self.ps:
                p.terminate()  # kill parallel environment workers
            # extra cleanup to find orphaned processes
            main_pgid = os.getpgid(os.getpid())
            for proc in psutil.process_iter():
                # if vizdoom or python process, belongs to this process group, and is not this (main) process
                # WARNING: this will kill concurrent training runs if they are launched from a common source (i.e. a bash script)
                # since they will share the same PGID
                if proc.name() in ('vizdoom', 'python') and os.getpgid(
                        proc.pid) == main_pgid and proc.pid != os.getpid():
                    proc.kill()
        else:
            if self.waiting:
                for remote in self.remotes:
                    remote.recv()
            for remote in self.remotes:
                remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode='human'):
        for remote in self.remotes:
            remote.send(('render', mode))
