import os
import time
import carla
import math
from loguru import logger
from datetime import datetime

import argparse
import random
import queue
import numpy as np
import carla_vehicle_process as cva
from WeatherSelector import WeatherSelector
import integrate_txt_file
import img_2_video

file_path = os.path.join(os.getcwd(), 'carla', 'generate_carla_data', 'multi_camera')
ROUTE = {12, 35, 36, 37, 38, 34, 2343, 2344, 2034, 2035}
IN_TOWN = {27, 13}

def retrieve_data(sensor_queue, frame, timeout=1):
    while True:
        try:
            data = sensor_queue.get(True, timeout)
        except queue.Empty:
            return None
        if data.frame == frame:
            return data


def set_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--host",
        default="127.0.0.1",
        type=str,
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "--port", default=2000, type=int, help="TCP port to listen to (default: 2000)"
    )
    argparser.add_argument(
        "--tm_port",
        default=8000,
        type=int,
        help="port to communicate with TM (default: 8000)",
    )
    argparser.add_argument(
        "--town_info_path",
        # default='/media/CityFlow/data/scenes',
        default="../scenes",
        type=str,
        help="path to town information",
    )
    argparser.add_argument(
        "--map_name", default="Town05", type=str, help="name of map: Town01-07"
    )
    argparser.add_argument(
        "--fps", default=0.05, type=float, help="fps of generated data"
    )
    argparser.add_argument(
        "--overlap",
        help="set whether the cemera filed of view overlaps",
        default=False,
    )
    argparser.add_argument(
        "--save_video", default=False, type=bool, help="generate video file"
    )
    argparser.add_argument(
        "--save_lidar", default=False, type=bool, help="save lidar images"
    )
    argparser.add_argument(
        "--number_of_vehicles",
        default=300,
        type=int,
        help="number of vehicles (default: 150)",
    )
    argparser.add_argument(
        "--number_walker", default=3, type=int, help="number of walker (default: 20)"
    )
    argparser.add_argument(
        "--weather_option",
        default=0,
        type=int,
        help="0: Day, 1: Dawn, 2: Rain, 4: Night",
    )
    argparser.add_argument(
        "--distance_between_v",
        default=2.0,
        type=float,
        help="distance between vehicles",
    )
    argparser.add_argument("--max_dist", default=120,
                           type=int, help="lidar range")
    # 没起作用，plan to set in code
    argparser.add_argument(
        "--resolution", default="720p", type=str, help="resolution of generated images"
    )
    argparser.add_argument(
        "--number_of_frame", default=2400, type=int, help="number of frames generated"
    )
    # TODO:！！！
    argparser.add_argument(
        "--number_of_dangerous_vehicles",
        default=50,
        type=int,
        help="number of dangerous_vehicles",
    )
    argparser.add_argument("--neptune", action="store_true")
    argparser.add_argument(
        "--output_path", default="..", type=str, help="path for output data"
    )
    args = argparser.parse_args()
    return args


def remove_unnecessary_objects(world):
    """Remove unuseful objects in the world, use opt maps for this function"""
    def remove_object(world,objs,obj):
        for ob in world.get_environment_objects(obj):
            objs.add(ob.id)
    world.unload_map_layer(carla.MapLayer.StreetLights)
    world.unload_map_layer(carla.MapLayer.Buildings)
    world.unload_map_layer(carla.MapLayer.Decals)
    world.unload_map_layer(carla.MapLayer.Walls)
    world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    # world.unload_map_layer(carla.MapLayer.Particles)
    world.unload_map_layer(carla.MapLayer.Ground)
    labels=[carla.CityObjectLabel.TrafficSigns,carla.CityObjectLabel.Other,
        carla.CityObjectLabel.Poles, carla.CityObjectLabel.Static,carla.CityObjectLabel.Dynamic,carla.CityObjectLabel.Buildings,
        carla.CityObjectLabel.Fences, carla.CityObjectLabel.Walls,carla.CityObjectLabel.Vegetation,carla.CityObjectLabel.Ground]
    objs = set()
    for label in labels:
        for obj in world.get_environment_objects(label):
            objs.add(obj.id)
    world.enable_environment_objects(objs, False)

def spawn_vehicle(transform, blueprint, traffic_manager, world):
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    # Taking out bicycles and motorcycles
    if int(blueprint.get_attribute("number_of_wheels")) > 2:
        if blueprint.has_attribute("color"):
            color = random.choice(
                blueprint.get_attribute("color").recommended_values
            )
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(
                blueprint.get_attribute("driver_id").recommended_values
            )
            # blueprint.set_attribute("driver_id", id)
        blueprint.set_attribute("role_name", "autopilot")

        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            actor.set_autopilot(enabled=True, tm_port=traffic_manager.get_port())
            traffic_manager.auto_lane_change(actor, True)
            traffic_manager.ignore_lights_percentage(actor, 100)
            traffic_manager.vehicle_percentage_speed_difference(actor, 40)
            traffic_manager.set_route(actor,
                                      ['Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight',
                                       'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight', 'Straight'])
        
            return actor.id
    
    return None

def main():
    args = set_args()
    logger.info(args)

    weather_dict = {
        0: "day",
        1: "dawn",
        2: "rain",
        3: "night",
    }

    vehicles_list = []
    nonvehicles_list = []
    num_cam = 0
    # set resolution
    image_resolution = {"720p": [1280, 720], "1080p": [1920, 1080],
                        "4k": [3840, 2160], "8k": [7680, 4320]}
    client = carla.Client(args.host, args.port)
    client.set_timeout(100.0)
    map_name = args.map_name+'_Opt'
    client.load_world(map_name)
    logger.info("***** Loading map *****")
    world = client.get_world()
    # remove parked vehicle
    world.unload_map_layer(carla.MapLayer.StreetLights)
    # world.unload_map_layer(carla.MapLayer.Buildings)
    world.unload_map_layer(carla.MapLayer.Decals)
    # world.unload_map_layer(carla.MapLayer.Walls)
    world.unload_map_layer(carla.MapLayer.Foliage)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    if args.overlap:
        town_info_path = os.path.join(file_path, args.output_path,  "scenes")
    elif not args.overlap:
        town_info_path = os.path.join(file_path, args.output_path, "scenes_non_overlap")

    town_path = f"{town_info_path}/{map_name}"
    weather_condition = weather_dict[args.weather_option]
    scence_path = f'{town_path}/{weather_condition}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'

    # Generate folders to save data
    logger.info("***** Generating folders *****")
    if not os.path.exists(os.path.dirname(scence_path)):
        os.makedirs(os.path.dirname(scence_path))

    files = os.listdir(f"{town_path}/camera_info")
    for file_name in files:
        if "camera" in file_name and file_name.endswith('txt'):
            num_cam += 1

    config_path = scence_path + "camera.txt"
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, "w") as config_file:
        for i in range(int(num_cam)):
            config_file.write(f"C0{i+1}\n")
        config_file.close()
    camera_data_path = []
    save_file = []
    for i in range(int(num_cam)):
        camera_data_path.append(f"{town_path}/camera_info/camera_{i + 1}.txt")
    for i in range(int(num_cam)):
        save_file.append(scence_path + "C%02d" % (i + 1))
        file_name = save_file[i] + "/"
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

    try:
        # ------------------------
        # Generate traffic manager
        # ------------------------
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(
            args.distance_between_v)
        # --------------
        # Set weather
        # --------------
        # weather_options = WeatherSelector().get_weather_options()
        # WeatherSelector.set_weather(world, weather_options[args.weather_option])

        # ClearNoon - works fine without reflection issues
        weather_options = WeatherSelector().get_weather_options()
        WeatherSelector.set_weather(
            world, weather_options[args.weather_option])
        # ---------------------
        # Set synchronous mode
        # ---------------------
        logger.info("***** RUNNING in synchronous mode *****")
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = args.fps
            world.apply_settings(settings)
        else:
            synchronous_master = False

        # set spectator
        spectator = world.get_spectator()
        spectator_location = carla.Location(x=0, y=0, z=0)
        spectator.set_transform(carla.Transform(spectator_location + carla.Location(z=400),
                                                carla.Rotation(roll=90, pitch=-90)))
        
        # town10 spectator: (30, -10, 250, pitch=-90)
        world.tick()
        # --------------
        # Get blueprints
        # --------------
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        blueprints = list(filter(lambda x: not(x.id.endswith('microlino') or 
                                               x.id.endswith('fusorosa') or
                                               x.id.endswith('firetruck') or
                                               x.id.endswith('cybertruck')), blueprints))
        blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
        map = world.get_map()
        spawn_points = map.get_spawn_points()

        # XXX: test code
        # STRAIGHT = {12, 35, 36}
        # CURVE = {37, 38, 34}
        # JUNCTION = {2344, 2035}
        spawn_transforms = spawn_points
        spawn_points = []
        spawn_points_34 = []
        spawn_points_38 = []
        for transform in spawn_transforms:
            wp = map.get_waypoint(transform.location)
            if wp.road_id is 12:
                spawn_points.append(transform)
                if wp.lane_id in {2, 3, 4}:
                    spawn_points_38.append(transform)
                if wp.lane_id in {-1, -2, -3}:
                    spawn_points_34.append(transform)
            # if wp.road_id in ROUTE:
            #     spawn_points.append(transform)
        # [world.debug.draw_point(point.location,size= 0.1, life_time=0.0) for point in spawn_points]
        # world.tick()
        
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = f"Requested {args.number_of_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points"
            logger.warning(msg, args.number_of_vehicles,
                           number_of_spawn_points)
            # args.number_of_vehicles = number_of_spawn_points

        # --------------
        # Spawn vehicles
        # --------------
        for i in range(args.number_of_vehicles):
            actor_id = spawn_vehicle(random.choice(spawn_points), random.choice(blueprints), traffic_manager, world)
            if actor_id is not None:
                vehicles_list.append(actor_id)
            [world.tick() for _ in range(4)]

        all_vehicles = world.get_actors().filter("vehicle.*")
        # set several of the cars as dangerous car
        for i in range(len(all_vehicles)):
            if i < args.number_of_dangerous_vehicles:
                danger_car = all_vehicles[i]
                # crazy car ignore traffic light, do not keep safe distance, and very fast
                # if i < args.number_of_dangerous_vehicles/4:
                #     traffic_manager.ignore_lights_percentage(danger_car, 100)
                traffic_manager.distance_to_leading_vehicle(danger_car, 5)
                # traffic_manager.auto_lane_change(danger_car, False)
                # traffic_manager.random_left_lanechange_percentage(
                #     danger_car, 30)
                # traffic_manager.random_right_lanechange_percentage(
                #     danger_car, 30)
                traffic_manager.vehicle_percentage_speed_difference(
                    danger_car, -200)
            else:
                normal_car = all_vehicles[i]
                # traffic_manager.auto_lane_change(normal_car, False)
                # traffic_manager.random_left_lanechange_percentage(
                #     normal_car, 5)
                # traffic_manager.random_right_lanechange_percentage(
                #     normal_car, 5)
                possible_speed_different = [0]
                # possible_speed_different = [-60, -90, -100, -130, -140]
                traffic_manager.vehicle_percentage_speed_difference(normal_car,
                                                                    possible_speed_different[i % len(possible_speed_different)])
        logger.info("Created %d vehicles" % len(vehicles_list))

        all_traffic_light = world.get_actors().filter("traffic.traffic_light*")

        # ban all traffic lights and set to green
        # for light in all_traffic_light:
        #     # set to green and freeze
        #     light.set_state(carla.TrafficLightState.Green)
        #     light.freeze(True)

        # set light time

        for light in all_traffic_light:
            light.set_red_time(15)
            light.set_green_time(15)

        # -------------
        # Spawn Walkers
        # -------------
        # 1. take all the random locations to spawn
        # walkers_list = []
        # walker_spawn_points = []
        # for i in range(args.number_walker):
        #     spawn_point = carla.Transform()
        #     loc = world.get_random_location_from_navigation()
        #     if loc != None:
        #         spawn_point.location = loc
        #         walker_spawn_points.append(spawn_point)

        # # 2. we spawn the walker object
        # batch_walker = []
        # for spawn_point in walker_spawn_points:
        #     walker_bp = random.choice(blueprintsWalkers)
        #     # set as not invincible
        #     if walker_bp.has_attribute("is_invincible"):
        #         walker_bp.set_attribute("is_invincible", "false")
        #     batch_walker.append(SpawnActor(walker_bp, spawn_point))

        # results = client.apply_batch_sync(batch_walker, True)
        # for i in range(len(results)):
        #     if results[i].error:
        #         logger.error(results[i].error)
        #     else:
        #         walkers_list.append({"id": results[i].actor_id})

        # # 3. we spawn the walker controller
        # batch_controller = []
        # walker_controller_bp = world.get_blueprint_library().find(
        #     "controller.ai.walker"
        # )
        # for i in range(len(walkers_list)):
        #     batch_controller.append(
        #         SpawnActor(
        #             walker_controller_bp, carla.Transform(
        #             ), walkers_list[i]["id"]
        #         )
        #     )
        # results = client.apply_batch_sync(batch_controller, True)
        # for i in range(len(results)):
        #     if results[i].error:
        #         logger.error(results[i].error)
        #     else:
        #         walkers_list[i]["con"] = results[i].actor_id

        # # 4. we put altogether the walkers and controllers id to get the objects from their id
        # all_id = []
        # for i in range(len(walkers_list)):
        #     all_id.append(walkers_list[i]["con"])
        #     all_id.append(walkers_list[i]["id"])

        # all_actors = world.get_actors(all_id)

        # # wait for a tick to ensure client receives the last transform of the walkers we have just created
        # # world.tick()
        # # world.wait_for_tick()

        # # 5. initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        # for i in range(0, len(all_id), 2):
        #     # start walker
        #     all_actors[i].start()
        #     # set walk to random point
        #     all_actors[i].go_to_location(
        #         world.get_random_location_from_navigation())
        #     # random max speed
        #     all_actors[i].set_max_speed(
        #         1 + random.random() / 2
        #     )  # max speed between 1 and 2 (default is 1.4 m/s)

        # logger.info("Created %d walkers \n" % len(walkers_list))
        # -----------------------------
        # Spawn sensors
        # -----------------------------
        q_list = []
        idx = 0
        tick_queue = queue.Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx = idx + 1
        # -----------------------------------------
        # Get camera position and spawn rgb camera
        # -----------------------------------------
        cam_bp_low_resolution = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp_low_resolution.set_attribute("sensor_tick", str(args.fps))
        # cam_bp.set_attribute('enable_postprocess_effects', str(False))
        cam_bp_low_resolution.set_attribute("image_size_x", str(
            image_resolution["720p"][0]))
        cam_bp_low_resolution.set_attribute("image_size_y", str(
            image_resolution["720p"][1]))

        cam_bp_middle_resolution = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp_middle_resolution.set_attribute("sensor_tick", str(args.fps))
        # cam_bp.set_attribute('enable_postprocess_effects', str(False))
        cam_bp_middle_resolution.set_attribute("image_size_x", str(
            image_resolution["720p"][0]))
        cam_bp_middle_resolution.set_attribute("image_size_y", str(
            image_resolution["720p"][1]))

        num_camera = len(camera_data_path)
        print('numer of camera: ', num_camera)
        for i in range(num_camera):
            cam_transform = cva.get_camera_position(camera_data_path[i])
            # draw camera position and id
            cva.draw_camera_position(camera_data_path[i], i+1, world)
            if i % 2 != 0:
                cam_transform.location.z = cam_transform.location.z
                cam = world.spawn_actor(
                    cam_bp_middle_resolution, cam_transform)
            else:
                cam_transform.location.z = cam_transform.location.z
                cam = world.spawn_actor(cam_bp_low_resolution, cam_transform)
            nonvehicles_list.append(cam)

            cam_queue = queue.Queue()
            cam.listen(cam_queue.put)
            q_list.append(cam_queue)
            cam_idx = idx
            idx = idx + 1

        logger.info("**** RGB camera ready ****")

        # --------------------
        # Spawn LIDAR sensor
        # --------------------

        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("sensor_tick", str(args.fps))
        lidar_bp.set_attribute("channels", "356")
        lidar_bp.set_attribute("points_per_second", "22400000")
        lidar_bp.set_attribute("upper_fov", "50")
        lidar_bp.set_attribute("lower_fov", "-50")
        lidar_bp.set_attribute("range", "180")
        lidar_bp.set_attribute("rotation_frequency", "40")

        for i in range(num_camera):
            cam_transform = cva.get_camera_position(camera_data_path[i])
            if i % 2 != 0:
                cam_transform.location.z = cam_transform.location.z
                lidar = world.spawn_actor(lidar_bp, cam_transform)
            else:
                cam_transform.location.z = cam_transform.location.z
                lidar = world.spawn_actor(lidar_bp, cam_transform)

            nonvehicles_list.append(lidar)
            lidar_queue = queue.Queue()
            lidar.listen(lidar_queue.put)
            q_list.append(lidar_queue)
            lidar_idx = idx
            idx = idx + 1

        logger.info("**** LIDAR ready ****")

        # -------------------
        # Spawn depth sensor
        # -------------------

        # depth_bp = world.get_blueprint_library().find("sensor.camera.depth")
        # depth_bp.set_attribute("sensor_tick", str(args.fps))
        # depth_bp.set_attribute(
        #     "image_size_x", str(image_resolution[args.resolution][0])
        # )
        # depth_bp.set_attribute(
        #     "image_size_y", str(image_resolution[args.resolution][1])
        # )
        # depth_bp.set_attribute("fov", "90")

        # for i in range(num_camera):
        #     cam_transform = cva.get_camera_position(camera_data_path[i])
        #     depth_camera = world.spawn_actor(depth_bp, cam_transform)
        #     nonvehicles_list.append(depth_camera)

        #     depth_queue = queue.Queue()
        #     depth_camera.listen(depth_queue.put)
        #     q_list.append(depth_queue)
        #     depth_idx = idx
        #     idx = idx + 1
        #     logger.info("**** Depth camera ready ****")

        # -------------------
        # Spawn segmentation sensor
        # -------------------
        # segm_bp = world.get_blueprint_library().find(
        #     "sensor.camera.semantic_segmentation"
        # )
        # segm_bp.set_attribute("sensor_tick", str(args.fps))
        # segm_bp.set_attribute("image_size_x", str(
        #     image_resolution[args.resolution][0]))
        # segm_bp.set_attribute("image_size_y", str(
        #     image_resolution[args.resolution][1]))
        # segm_bp.set_attribute("fov", "90")

        # iseg_bp = world.get_blueprint_library().find(
        #     "sensor.camera.instance_segmentation"
        # )
        # iseg_bp.set_attribute("sensor_tick", str(args.fps))
        # iseg_bp.set_attribute("image_size_x", str(
        #     image_resolution[args.resolution][0]))
        # iseg_bp.set_attribute("image_size_y", str(
        #     image_resolution[args.resolution][1]))
        # iseg_bp.set_attribute("fov", "90")

        # for i in range(num_camera):
        #     cam_transform = cva.get_camera_position(camera_data_path[i])
        #     segm_camera = world.spawn_actor(segm_bp, cam_transform)
        #     nonvehicles_list.append(segm_camera)

        #     seg_queue = queue.Queue()
        #     segm_camera.listen(seg_queue.put)
        #     q_list.append(seg_queue)
        #     segm_idx = idx
        #     idx = idx + 1
        #     logger.info("**** Semantic Segmentation camera ready ****")

        # for i in range(num_camera):
        #     cam_transform = cva.get_camera_position(camera_data_path[i])
        #     iseg_camera = world.spawn_actor(iseg_bp, cam_transform)
        #     nonvehicles_list.append(iseg_camera)

        #     iseg_queue = queue.Queue()
        #     iseg_camera.listen(iseg_queue.put)
        #     q_list.append(iseg_queue)
        #     iseg_idx = idx
        #     idx = idx + 1
        #     logger.info("**** Instance Segmentation camera ready ****")

        # ---------------
        # Begin the loop
        # ---------------
        time_sim = 0
        frame_number = 0
        save_depth = False
        save_segm = False
        logs_path = os.path.join(scence_path, "logs.txt")
        logger.info("**** Begin the loop ****")
        vehicles_raw = None
        while True:
            if frame_number == args.number_of_frame:
                break
            # Extract the available data
            if vehicles_raw is None:
                vehicles_raw = world.get_actors().filter("vehicle.*")
            else:
                for vehicle in vehicles_raw.filter("*"):
                    if vehicle.is_alive:
                        wp = map.get_waypoint(vehicle.get_location())
                        if wp.road_id is 38 and wp.lane_id in {-1, -2, -3} and wp.s > 280.0:
                            vehicle.destroy()
                            vehicles_list.remove(vehicle.id)
                        if wp.road_id is 34 and wp.lane_id in {2, 3, 4} and wp.s < 20.0:
                            vehicle.destroy()
                            vehicles_list.remove(vehicle.id)

            if(len(vehicles_raw) < args.number_of_vehicles):                
                right_actor = spawn_vehicle(random.choice(spawn_points_34), random.choice(blueprints), traffic_manager, world)
                left_actor = spawn_vehicle(random.choice(spawn_points_38), random.choice(blueprints), traffic_manager, world)
                if right_actor is not None:
                    vehicles_list.append(right_actor)
                if left_actor is not None:
                    vehicles_list.append(left_actor)
                
            t1 = time.time()
            nowFrame = world.tick()

            # Check whether it's time for sensor to capture data
            if time_sim >= 0.1:
                data = [retrieve_data(q, nowFrame) for q in q_list]
                assert all(x.frame == nowFrame for x in data if x is not None)

                # Skip if any sensor data is not available
                if None in data:
                    continue

                frame_number += 1

                vehicles_raw = world.get_actors().filter("vehicle.*")
                walker_raw = world.get_actors().filter("walker.*")
                snap = data[tick_idx]
                if save_depth and save_segm:
                    rgb_img = data[cam_idx - num_camera +
                                   1: lidar_idx - num_camera + 1]
                    lidar_img = data[
                        lidar_idx - num_camera + 1: depth_idx - num_camera + 1
                    ]
                    depth_img = data[depth_idx - num_camera +
                                     1: segm_idx - num_camera + 1]
                    segm_img = data[segm_idx - num_camera +
                                    1: iseg_idx - num_camera + 1]
                    iseg_img = data[iseg_idx - num_camera + 1:]
                else:
                    rgb_img = data[cam_idx - num_camera +
                                   1: lidar_idx - num_camera + 1]
                    lidar_img = data[lidar_idx - num_camera + 1:]

                # Attach additional information to the snapshot
                vehicles = cva.snap_processing(vehicles_raw, walker_raw, snap)

                # Calculating visible bounding boxesw
                for i in range(num_camera):
                    cam = nonvehicles_list[i]
                    Lidar_img = lidar_img[i]
                    Rgb_img = rgb_img[i]

                    if save_depth:
                        Depth_img = depth_img[i]
                        save_path = save_file[i] + "/out_depth"
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        Depth_img.save_to_disk(
                            save_path + "/%06d.jpg" % frame_number,
                            carla.ColorConverter.LogarithmicDepth,
                        )

                    if save_segm:
                        Segm_img = segm_img[i]
                        Iseg_img = iseg_img[i]
                        save_path = save_file[i] + "/out_segm"
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        Segm_img.save_to_disk(
                            save_path + "/%06d.png" % frame_number,
                            carla.ColorConverter.CityScapesPalette,
                        )
                        save_path = save_file[i] + "/out_iseg"
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        Iseg_img.save_to_disk(
                            save_path + "/%06d.png" % frame_number,
                        )

                    filtered_out, _ = cva.get_all_data(
                        vehicles,
                        cam,
                        Lidar_img,
                        show_img=Rgb_img,
                        json_path=os.path.join(
                            file_path, "vehicle_class_json_file.txt"),
                        path=save_file[i],
                        framenumber=frame_number,
                        max_dist=args.max_dist,
                        save_lidar=args.save_lidar,
                    )
                    # Save the results
                    cva.save_output(
                        frame_number,
                        Rgb_img,
                        filtered_out["vehicles_id"],
                        filtered_out["bbox"],
                        filtered_out["bbox_3d"],
                        filtered_out["world_coords"],
                        filtered_out["class"],
                        save_patched=False,
                        out_format="json",
                        path=save_file[i],
                    )
                    spend_time = time.time() - t1

                logger.info(
                    "Generate frame: %d  Spend time: %d s" % (
                        frame_number, spend_time)
                )

                with open(logs_path, "a+") as f:
                    f.write(f"Frame:{frame_number}, Spend time:{spend_time}\n")

                time_sim = 0
            time_sim = time_sim + settings.fixed_delta_seconds

    finally:
        try:
            for nonvehicle in nonvehicles_list:
                nonvehicle.stop()
        except:
            logger.info("Sensors has not been initiated")

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        logger.info("Destroying %d vehicles" % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x)
                           for x in vehicles_list])

        # logger.info("Destroying %d NPC walkers" % len(walkers_list))
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        # for i in range(0, len(all_id), 2):
        #     all_actors[i].stop()
        # client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        logger.info("Destroying %d nonvehicles" % len(nonvehicles_list))
        client.apply_batch([carla.command.DestroyActor(x)
                           for x in nonvehicles_list])

        # ---------------------------------------
        # Begin the Processing the generate data
        # ---------------------------------------

        if args.save_video:
            logger.info("Generating the Video...")
            img_2_video.img2video(save_file)
        time.sleep(2.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Finish!")
