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
# Intersection
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


# ----------------------------
# Shadow check
# ----------------------------

def in_shadow(point, normal, light, scene):
    eps = 1e-4
    shadow_origin = point + eps * normal
    light_dir = normalize(light["position"] - point)

    light_dist = np.linalg.norm(light["position"] - point)

    for sphere in scene["spheres"]:
        t = ray_sphere_intersect(shadow_origin, light_dir, sphere)
        if t is not None and t < light_dist:
            return True

    return False


# ----------------------------
# Phong shading (unchanged)
# ----------------------------

def phong(point, normal, view_dir, sphere, light):
    light_dir = normalize(light["position"] - point)

    ambient = light["ambient"] * sphere["color"]

    diff = max(np.dot(normal, light_dir), 0.0)
    diffuse = light["diffuse"] * diff * sphere["color"]

    reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
    spec = max(np.dot(view_dir, reflect_dir), 0.0) ** sphere["shininess"]
    specular = light["specular"] * spec * light["color"]

    return ambient, diffuse, specular


# ----------------------------
# Main ray tracing (RECURSIVE)
# ----------------------------

def trace_ray(ray_origin, ray_dir, scene, camera, depth):
    max_depth = 3

    closest_t = float("inf")
    hit_sphere = None

    for sphere in scene["spheres"]:
        t = ray_sphere_intersect(ray_origin, ray_dir, sphere)
        if t is not None and t < closest_t:
            closest_t = t
            hit_sphere = sphere

    if hit_sphere is None:
        return scene["background"]

    # Hit point
    hit_point = ray_origin + closest_t * ray_dir
    normal = normalize(hit_point - hit_sphere["center"])
    view_dir = normalize(camera["position"] - hit_point)

    # --- SHADOW CHECK ---
    shadow = in_shadow(hit_point, normal, scene["light"], scene)

    ambient, diffuse, specular = phong(
        hit_point, normal, view_dir, hit_sphere, scene["light"]
    )

    if shadow:
        color = ambient  # only ambient if in shadow
    else:
        color = ambient + diffuse + specular

    # --- REFLECTION ---
    if depth < max_depth and hit_sphere.get("reflectivity", 0) > 0:
        reflect_dir = ray_dir - 2 * np.dot(ray_dir, normal) * normal
        reflect_origin = hit_point + 1e-4 * normal

        reflected_color = trace_ray(
            reflect_origin, normalize(reflect_dir), scene, camera, depth + 1
        )

        color = (1 - hit_sphere["reflectivity"]) * color + \
                hit_sphere["reflectivity"] * reflected_color

    return np.clip(color, 0, 1)


# ----------------------------
# Rendering
# ----------------------------

def render(scene, camera, width, height):
    image = np.zeros((height, width, 3))

    aspect_ratio = width / height
    viewport_height = 2.0
    viewport_width = viewport_height * aspect_ratio

    origin = camera["position"]

    for j in range(height):
        for i in range(width):
            u = (i + 0.5) / width
            v = (j + 0.5) / height

            x = (2 * u - 1) * (viewport_width / 2)
            y = (1 - 2 * v) * (viewport_height / 2)
            z = -1.0

            pixel = np.array([x, y, z])
            ray_dir = normalize(pixel - origin)

            color = trace_ray(origin, ray_dir, scene, camera, depth=0)
            image[j, i] = color

    return image


# ----------------------------
# Main
# ----------------------------

def main():
    width = 800
    height = 600

    camera = {
        "position": np.array([0.0, 0.0, 0.0])
    }

    scene = {
        "background": np.array([1.0, 1.0, 1.0]),

        "light": {
            "position": np.array([5.0, 5.0, 0.0]),
            "color": np.array([1.0, 1.0, 1.0]),
            "ambient": 0.1,
            "diffuse": 0.7,
            "specular": 0.4
        },

        "spheres": [
            {
                "center": np.array([0.0, 0.0, -3.5]),
                "radius": 0.9,
                "color": np.array([1.0, 0.2, 0.2]),
                "shininess": 32,
                "reflectivity": 0.3
            },
            {
                "center": np.array([1.4, -0.3, -4.5]),
                "radius": 1.0,
                "color": np.array([0.2, 0.8, 0.3]),
                "shininess": 16,
                "reflectivity": 0.2
            },
            {
                "center": np.array([-1.5, 0.4, -4.0]),
                "radius": 0.7,
                "color": np.array([0.2, 0.4, 1.0]),
                "shininess": 64,
                "reflectivity": 0.5
            },
        ]
    }

    image = render(scene, camera, width, height)

    plt.imshow(image)
    plt.axis("off")
    plt.title("Step 3: Shadows + Reflections")
    plt.show()

    plt.imsave("step3.png", image)


if __name__ == "__main__":
    main()