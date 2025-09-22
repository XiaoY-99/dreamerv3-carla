import carla

def build_loop_route(carla_map: carla.Map, step_m: float = 2.0, max_points: int = 2000):
    seed_wp = carla_map.get_waypoint(carla_map.get_spawn_points()[0].location)
    route = [seed_wp]
    wp = seed_wp
    for _ in range(max_points):
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp = nxt[0]
        route.append(wp)
        if wp.transform.location.distance(seed_wp.transform.location) < 2.0 and len(route) > 12:
            break
    return route
