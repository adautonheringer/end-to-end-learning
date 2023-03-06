#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import numpy.random as random
import re
import sys
import weakref
import random as rd

from keras.models import load_model

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    os.environ["DISPLAY"]
except:
    os.environ["SDL_VIDEODRIVER "] = "dummy"

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import time

from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.local_planner import LocalPlanner


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def add_noise(vehicle):
    location = vehicle.get_location()
    transform = vehicle.get_transform()
    rotation = transform.rotation

    vehicle.set_transform(
        carla.Transform(
            carla.Location(x=location.x + rd.randint(-1, 1), y=location.y + rd.randint(-1, 1), z=location.z),
            carla.Rotation(pitch=rotation.pitch, yaw=rotation.yaw + rd.randint(-30, 30), roll=rotation.roll)))


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        # Town02 Longer Route
        self.location_1 = carla.Location(x=-7.530000, y=121.209999, z=0.500000)
        self.location_2 = carla.Location(x=-7.530004, y=234.110001, z=0.500000)
        self.location_3 = carla.Location(x=166.914505, y=191.770035, z=0.500000)
        self.location_4 = carla.Location(x=173.870056, y=105.550011, z=0.500000)

        self.transform_4 = carla.Transform(self.location_4, carla.Rotation(yaw=-180))
        self.transform_1 = carla.Transform(self.location_1, carla.Rotation(yaw=90))
        self.transform_2 = carla.Transform(self.location_2, carla.Rotation(yaw=90))
        self.transform_3 = carla.Transform(self.location_3, carla.Rotation(yaw=0))

        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            # spawn_points = self.map.get_spawn_points()
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            # self.player = self.world.try_spawn_actor(blueprint, spawn_point)

            self.player = self.world.try_spawn_actor(blueprint, self.transform_1)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor['center'],
            self.camera_manager.sensor['left'],
            self.camera_manager.sensor['right'],
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                x = actor.destroy()
                if x:
                    print(f'actor {actor} destroyed')


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x) ** 2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z) ** 2)

        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    ####
                    # if item[0] == 'Throttle:':
                    #     display.blit(
                    #         self._font_mono.render(f'Throttle: {item[1]}', True, (255, 255, 255)),
                    #         (150, v_offset + 16))

                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = {}
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.frame = None
        self.hlc = None
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                blp.set_attribute('sensor_tick', '0.1')
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, agent: BasicAgent = None, notify=True, force_respawn=False,
                   traffic_manager: carla.TrafficManager = None):

        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
                force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if True:
            # if self.sensor is not None:
            #     self.sensor.destroy()
            #     self.surface = None
            self.sensor['center'] = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                carla.Transform(carla.Location(x=1.6, z=1.7)),
                attach_to=self._parent)

            self.sensor['left'] = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                carla.Transform(carla.Location(x=1.6, z=1.7, y=-1)),
                attach_to=self._parent)

            self.sensor['far_left'] = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                carla.Transform(carla.Location(x=1.6, z=1.7, y=-2)),
                attach_to=self._parent)

            self.sensor['far_far_left'] = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                carla.Transform(carla.Location(x=1.6, z=1.7, y=-3)),
                attach_to=self._parent)

            self.sensor['right'] = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                carla.Transform(carla.Location(x=1.6, z=1.7, y=1)),
                attach_to=self._parent)

            self.sensor['far_right'] = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                carla.Transform(carla.Location(x=1.6, z=1.7, y=2)),
                attach_to=self._parent)

            self.sensor['far_far_right'] = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                carla.Transform(carla.Location(x=1.6, z=1.7, y=3)),
                attach_to=self._parent)

            # self.sensor['yaw_left'] = self._parent.get_world().spawn_actor(
            #     self.sensors[index][-1],
            #     carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=-10)),
            #     attach_to=self._parent)
            #
            # self.sensor['yaw_far_left'] = self._parent.get_world().spawn_actor(
            #     self.sensors[index][-1],
            #     carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=-20)),
            #     attach_to=self._parent)
            #
            # self.sensor['yaw_far_far_left'] = self._parent.get_world().spawn_actor(
            #     self.sensors[index][-1],
            #     carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=-30)),
            #     attach_to=self._parent)
            #
            # self.sensor['yaw_far_far_far_left'] = self._parent.get_world().spawn_actor(
            #     self.sensors[index][-1],
            #     carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=-40)),
            #     attach_to=self._parent)

            # self.sensor['yaw_right'] = self._parent.get_world().spawn_actor(
            #     self.sensors[index][-1],
            #     carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=10)),
            #     attach_to=self._parent)
            #
            # self.sensor['yaw_far_right'] = self._parent.get_world().spawn_actor(
            #     self.sensors[index][-1],
            #     carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=20)),
            #     attach_to=self._parent)
            #
            # self.sensor['yaw_far_far_right'] = self._parent.get_world().spawn_actor(
            #     self.sensors[index][-1],
            #     carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=30)),
            #     attach_to=self._parent)
            #
            # self.sensor['yaw_far_far_far_right'] = self._parent.get_world().spawn_actor(
            #     self.sensors[index][-1],
            #     carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=40)),
            #     attach_to=self._parent)


            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor['center'].listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'center', agent, traffic_manager))
            self.sensor['left'].listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'left', agent, traffic_manager))
            self.sensor['far_left'].listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'far_left', agent, traffic_manager))
            self.sensor['far_far_left'].listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'far_far_left', agent, traffic_manager))
            self.sensor['right'].listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'right', agent, traffic_manager))
            self.sensor['far_right'].listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'far_right', agent, traffic_manager))
            self.sensor['far_far_right'].listen(
                lambda image: CameraManager._parse_image(weak_self, image, 'far_far_right', agent, traffic_manager))

            # self.sensor['yaw_left'].listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'yaw_left', agent, traffic_manager))
            # self.sensor['yaw_far_left'].listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'yaw_far_left', agent, traffic_manager))
            # self.sensor['yaw_far_far_left'].listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'yaw_far_far_left', agent, traffic_manager))
            # self.sensor['yaw_far_far_far_left'].listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'yaw_far_far_far_left', agent, traffic_manager))
            #
            # self.sensor['yaw_right'].listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'yaw_right', agent, traffic_manager))
            # self.sensor['yaw_far_right'].listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'yaw_far_right', agent, traffic_manager))
            # self.sensor['yaw_far_far_right'].listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'yaw_far_far_right', agent, traffic_manager))
            # self.sensor['yaw_far_far_far_right'].listen(
            #     lambda image: CameraManager._parse_image(weak_self, image, 'yaw_far_far_far_right', agent, traffic_manager))



        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, position: str, agent: BasicAgent, traffic_manager: carla.TrafficManager):
        self = weak_self()
        array = None
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            if position == 'center':
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            control = self._parent.get_control()
            if position == 'left':
                control.steer = control.steer + 0.2
            elif position == 'far_left':
                control.steer = control.steer + 0.35
            elif position == 'far_far_left':
                control.steer = control.steer + 0.60
            elif position == 'right':
                control.steer = control.steer - 0.2
            elif position == 'far_right':
                control.steer = control.steer - 0.35
            elif position == 'far_far_right':
                control.steer = control.steer - 0.6

            # elif position == 'yaw_left':
            #     control.steer = control.steer + 0.2
            # elif position == 'yaw_far_left':
            #     control.steer = control.steer + 0.4
            # elif position == 'yaw_far_far_left':
            #     control.steer = control.steer + 0.6
            # elif position == 'yaw_far_far_far_left':
            #     control.steer = control.steer + 0.8
            #
            # elif position == 'yaw_right':
            #     control.steer = control.steer - 0.2
            # elif position == 'yaw_far_right':
            #     control.steer = control.steer - 0.4
            # elif position == 'yaw_far_far_right':
            #     control.steer = control.steer - 0.6
            # elif position == 'yaw_far_far_far_right':
            #     control.steer = control.steer - 0.8

            # high_level = agent.get_local_planner().get_incoming_waypoint_and_direction(0)[1].value
            if traffic_manager is not None:
                high_level = traffic_manager.get_next_action(self._parent)[0]
                self.hlc = high_level
                self.frame = image.frame
                vel = self._parent.get_velocity()
                current_speed = (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))
                speed_limit = self._parent.get_speed_limit()

                label = np.array(
                    [control.steer, control.throttle, control.brake, high_level, current_speed, speed_limit])
                np.savez(f'../dataset_2/%08d_{position}' % image.frame, x=array, y=label)
                # image.save_to_disk(f'_out/%08d_{position}' % image.frame)


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    pygame.init()
    pygame.font.init()
    world = None

    weather_preset = [carla.WeatherParameters.ClearNoon,
                      carla.WeatherParameters.CloudyNoon,
                      carla.WeatherParameters.WetNoon,
                      carla.WeatherParameters.WetCloudyNoon,
                      carla.WeatherParameters.ClearSunset,
                      carla.WeatherParameters.CloudySunset,
                      carla.WeatherParameters.WetSunset,
                      carla.WeatherParameters.WetCloudySunset]

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        client.load_world('Town02')

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        # display = pygame.display.set_mode(
        #     (args.width, args.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world)
        opt_dict = {"ignore_traffic_lights": True, "ignore_stop_signs": True}
        if args.agent == "Basic":
            agent = BasicAgent(vehicle=world.player, target_speed=15, opt_dict=opt_dict)
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        spectator = sim_world.get_spectator()
        transform = carla.Transform(carla.Location(90, 180, 160), carla.Rotation(pitch=-90))
        spectator.set_transform(transform)

        locations = [world.location_1, world.location_2, world.location_3, world.location_4]
        commands = ["Straight", "Left", "Right", "Left", "Right", "Left"]
        path_list = []
        command_list = []
        for i in range(100):
            path_list += locations
            command_list += commands
        # traffic_manager.set_path(world.player, path_list)
        traffic_manager.set_route(world.player, command_list)

        clock = pygame.time.Clock()

        camera_manager = CameraManager(world.player, hud)
        camera_manager.set_sensor(0, agent, traffic_manager=traffic_manager)
        camera_manager.toggle_recording()

        count = 0
        noise_counter = 0
        sim_world.set_weather(weather_preset[rd.randint(0, 7)])
        # sim_world.set_weather(carla.WeatherParameters.ClearSunset)
        traffic_manager.ignore_lights_percentage(world.player, 100)
        world.player.set_autopilot(True)

        while True:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            world.tick(clock)
            # world.render(display)
            # pygame.display.flip()

            # if agent.done():
            #     if args.loop:
            #         camera_manager.toggle_recording()
            #
            #         if where_to == world.start_location:
            #             count += 1
            #             print(f"{count} turn finished")
            #             sim_world.set_weather(weather_preset[rd.randint(0, 13)])
            #             where_to = world.end_location
            #         else:
            #             where_to = world.start_location
            #
            #         agent.set_destination(where_to)
            #
            #         if count == 20:
            #             print("The target has been reached, stopping the simulation")
            #             break
            #
            #         world.hud.notification("The target has been reached, searching for another target", seconds=4.0)
            #         # print("The target has been reached, searching for another target")
            #     else:
            #         print("The target has been reached, stopping the simulation")
            #         break

            # noise_counter += 1
            # if noise_counter % 200 == 0:
            #     add_noise(world.player)
            #     print(f"TIME: {hud.simulation_time}")
            #     print(f"FPS: {hud.server_fps}")
            vel = world.player.get_velocity()
            current_speed = (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))

            print(f"TIME: {hud.simulation_time} and SPEED: {current_speed} and INTENTION: {camera_manager.hlc} and FRAME: {camera_manager.frame}")
            noise_counter += 1
            if noise_counter % 500 == 0:
                weather = weather_preset[rd.randint(0, 7)]
                sim_world.set_weather(weather)
                print(f"chosen weather: {weather}")

            if hud.simulation_time > 1810:
                break

            # control = agent.run_step()
            # control.manual_gear_shift = False
            # control.throttle = min(0.5, control.throttle)
            # world.player.apply_control(control)

    finally:

        if world is not None:
            camera_manager.sensor['center'].stop()
            camera_manager.sensor['left'].stop()
            camera_manager.sensor['right'].stop()

            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(False)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='192.168.5.145',
        help='IP of the host server (default: 192.168.5.145)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='300x180',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_false',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.lincoln.mkz_2020',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_false',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Basic")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
