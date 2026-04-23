import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Math helpers
# ----------------------------

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# ----------------------------
# Camera
# ----------------------------

def make_orbit_camera(target, radius, azimuth_deg, elevation, fov_deg=45):
    az = np.radians(azimuth_deg)

    position = np.array([
        target[0] + radius * np.cos(az),
        target[1] + elevation,
        target[2] + radius * np.sin(az)
    ], dtype=np.float32)

    forward = normalize(target - position)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))

    return {
        "position": position,
        "target": target,
        "forward": forward,
        "right": right,
        "up": up,
        "fov_deg": fov_deg
    }


# ----------------------------
# Intersections
# ----------------------------

def ray_sphere_intersect(ray_origin, ray_dir, sphere):
    center = sphere["center"]
    radius = sphere["radius"]

    oc = ray_origin - center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    eps = 1e-6
    valid_ts = [t for t in (t1, t2) if t > eps]
    return min(valid_ts) if valid_ts else None


def ray_annulus_intersect(ray_origin, ray_dir, annulus):
    """
    Intersect ray with an annulus (ring) lying in a plane.
    """
    center = annulus["center"]
    normal = normalize(annulus["normal"])

    denom = np.dot(ray_dir, normal)
    eps = 1e-6

    if abs(denom) < eps:
        return None

    t = np.dot(center - ray_origin, normal) / denom
    if t <= eps:
        return None

    hit_point = ray_origin + t * ray_dir
    radial_vec = hit_point - center

    # remove any component along the normal, leaving in-plane radius
    radial_vec = radial_vec - np.dot(radial_vec, normal) * normal
    r = np.linalg.norm(radial_vec)

    if annulus["inner_radius"] <= r <= annulus["outer_radius"]:
        return t

    return None


# ----------------------------
# Starfield background
# ----------------------------

def hash_direction(d):
    """
    Deterministic pseudo-random number based on ray direction.
    """
    x = np.sin(np.dot(d, np.array([12.9898, 78.233, 37.719])))
    return x - np.floor(x)


def starfield(ray_dir):
    """
    Simple procedural starfield.
    """
    d = normalize(ray_dir)

    # dark blue space gradient
    t = 0.5 * (d[1] + 1.0)
    base = (1.0 - t) * np.array([0.01, 0.01, 0.03]) + t * np.array([0.0, 0.0, 0.01])

    # sparse stars
    h = hash_direction(np.floor(d * 250.0))
    if h > 0.997:
        brightness = 0.7 + 0.3 * hash_direction(np.floor(d * 500.0))
        star_color = brightness * np.array([1.0, 1.0, 1.0])
        return np.clip(base + star_color, 0, 1)

    return base


# ----------------------------
# Disk appearance
# ----------------------------

def disk_color(hit_point, annulus):
    """
    Give the accretion disk a warm radial glow.
    """
    center = annulus["center"]
    normal = normalize(annulus["normal"])

    radial_vec = hit_point - center
    radial_vec = radial_vec - np.dot(radial_vec, normal) * normal
    r = np.linalg.norm(radial_vec)

    r0 = annulus["inner_radius"]
    r1 = annulus["outer_radius"]
    u = (r - r0) / (r1 - r0)

    # brighter near inner edge, darker toward outer edge
    inner = np.array([1.0, 0.85, 0.45])
    mid   = np.array([1.0, 0.45, 0.15])
    outer = np.array([0.45, 0.08, 0.02])

    if u < 0.4:
        a = u / 0.4
        color = (1 - a) * inner + a * mid
    else:
        a = (u - 0.4) / 0.6
        color = (1 - a) * mid + a * outer

    return np.clip(color, 0, 1)


# ----------------------------
# Scene tracing
# ----------------------------

def trace_ray(ray_origin, ray_dir, scene):
    closest_t = float("inf")
    hit_type = None
    hit_obj = None

    # black hole sphere
    for sphere in scene["spheres"]:
        t = ray_sphere_intersect(ray_origin, ray_dir, sphere)
        if t is not None and t < closest_t:
            closest_t = t
            hit_type = "sphere"
            hit_obj = sphere

    # annulus / disk
    for annulus in scene["annuli"]:
        t = ray_annulus_intersect(ray_origin, ray_dir, annulus)
        if t is not None and t < closest_t:
            closest_t = t
            hit_type = "annulus"
            hit_obj = annulus

    if hit_type is None:
        return starfield(ray_dir)

    hit_point = ray_origin + closest_t * ray_dir

    if hit_type == "sphere":
        # placeholder black hole: pure black
        return hit_obj["color"]

    if hit_type == "annulus":
        return disk_color(hit_point, hit_obj)

    return starfield(ray_dir)


# ----------------------------
# Rendering
# ----------------------------

def render(scene, camera, width, height):
    image = np.zeros((height, width, 3), dtype=np.float32)

    aspect_ratio = width / height
    fov = np.radians(camera["fov_deg"])
    scale = np.tan(fov / 2)

    origin = camera["position"]
    forward = camera["forward"]
    right = camera["right"]
    up = camera["up"]

    for j in range(height):
        for i in range(width):
            px = (2 * ((i + 0.5) / width) - 1) * aspect_ratio * scale
            py = (1 - 2 * ((j + 0.5) / height)) * scale

            ray_dir = normalize(forward + px * right + py * up)
            image[j, i] = trace_ray(origin, ray_dir, scene)

    return image


# ----------------------------
# Main
# ----------------------------

def main():
    width = 1000
    height = 700

    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    camera = make_orbit_camera(
        target=target,
        radius=8.0,
        azimuth_deg=35.0,
        elevation=2.0,
        fov_deg=40.0
    )

    scene = {
        "spheres": [
            {
                "center": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "radius": 1.0,
                "color": np.array([0.0, 0.0, 0.0], dtype=np.float32)  # black hole placeholder
            }
        ],
        "annuli": [
            {
                "center": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "normal": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "inner_radius": 1.3,
                "outer_radius": 3.0
            }
        ]
    }

    image = render(scene, camera, width, height)

    plt.figure(figsize=(12, 8))
    plt.imshow(np.clip(image, 0, 1))
    plt.axis("off")
    plt.title("Step 4: Space Scene Setup")
    plt.show()

    plt.imsave("step4_space_scene.png", np.clip(image, 0, 1))


if __name__ == "__main__":
    main()