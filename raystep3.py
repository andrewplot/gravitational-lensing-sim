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


def find_nearest_hit(ray_origin, ray_dir, scene):
    closest_t = float("inf")
    hit_sphere = None

    for sphere in scene["spheres"]:
        t = ray_sphere_intersect(ray_origin, ray_dir, sphere)
        if t is not None and t < closest_t:
            closest_t = t
            hit_sphere = sphere

    if hit_sphere is None:
        return None, None

    return hit_sphere, closest_t


# ----------------------------
# Shadow check
# ----------------------------

def in_shadow(point, normal, light, scene, current_sphere):
    eps = 1e-4
    shadow_origin = point + eps * normal
    light_vec = light["position"] - shadow_origin
    light_dir = normalize(light_vec)
    light_dist = np.linalg.norm(light_vec)

    for sphere in scene["spheres"]:
        if sphere is current_sphere:
            continue

        t = ray_sphere_intersect(shadow_origin, light_dir, sphere)
        if t is not None and t < light_dist:
            return True

    return False


# ----------------------------
# Phong shading
# ----------------------------

def phong_components(point, normal, view_dir, sphere, light):
    light_dir = normalize(light["position"] - point)

    # Ambient
    ambient = light["ambient"] * sphere["color"]

    # Diffuse
    ndotl = max(np.dot(normal, light_dir), 0.0)
    diffuse = light["diffuse"] * ndotl * sphere["color"]

    # Specular
    reflect_light = 2 * np.dot(normal, light_dir) * normal - light_dir
    spec_angle = max(np.dot(view_dir, normalize(reflect_light)), 0.0)
    specular = light["specular"] * (spec_angle ** sphere["shininess"]) * light["color"]

    return ambient, diffuse, specular


# ----------------------------
# Recursive ray tracing
# ----------------------------

def trace_ray(ray_origin, ray_dir, scene, depth=0):
    max_depth = 3

    hit_sphere, t = find_nearest_hit(ray_origin, ray_dir, scene)

    if hit_sphere is None:
        return scene["background"]

    hit_point = ray_origin + t * ray_dir
    normal = normalize(hit_point - hit_sphere["center"])

    # More correct than using camera position once recursion starts
    view_dir = normalize(-ray_dir)

    ambient, diffuse, specular = phong_components(
        hit_point, normal, view_dir, hit_sphere, scene["light"]
    )

    shadowed = in_shadow(hit_point, normal, scene["light"], scene, hit_sphere)

    if shadowed:
        local_color = ambient
    else:
        local_color = ambient + diffuse + specular

    color = local_color.copy()

    # Reflection
    reflectivity = hit_sphere.get("reflectivity", 0.0)
    if depth < max_depth and reflectivity > 0.0:
        reflect_dir = normalize(ray_dir - 2 * np.dot(ray_dir, normal) * normal)
        reflect_origin = hit_point + 1e-4 * normal

        reflected_color = trace_ray(reflect_origin, reflect_dir, scene, depth + 1)

        # Add reflection instead of blending away the whole object
        color = color + reflectivity * reflected_color

    return np.clip(color, 0, 1)


# ----------------------------
# Rendering
# ----------------------------

def render(scene, camera, width, height):
    image = np.zeros((height, width, 3), dtype=np.float32)

    aspect_ratio = width / height
    viewport_height = 2.0
    viewport_width = viewport_height * aspect_ratio

    origin = camera["position"]
    image_plane_z = -1.0

    for j in range(height):
        for i in range(width):
            u = (i + 0.5) / width
            v = (j + 0.5) / height

            x = (2 * u - 1) * (viewport_width / 2)
            y = (1 - 2 * v) * (viewport_height / 2)
            z = image_plane_z

            pixel_pos = np.array([x, y, z], dtype=np.float32)
            ray_dir = normalize(pixel_pos - origin)

            image[j, i] = trace_ray(origin, ray_dir, scene, depth=0)

    return image


# ----------------------------
# Main
# ----------------------------

def main():
    width = 800
    height = 600

    camera = {
        "position": np.array([0.0, 0.0, 0.0], dtype=np.float32)
    }

    scene = {
        "background": np.array([1.0, 1.0, 1.0], dtype=np.float32),  # white test background

        "light": {
            "position": np.array([5.0, 5.0, 0.0], dtype=np.float32),
            "color": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "ambient": 0.12,
            "diffuse": 0.7,
            "specular": 0.35
        },

        "spheres": [
            {
                "center": np.array([0.0, 0.0, -3.5], dtype=np.float32),
                "radius": 0.9,
                "color": np.array([1.0, 0.2, 0.2], dtype=np.float32),
                "shininess": 32,
                "reflectivity": 0.08
            },
            {
                "center": np.array([1.4, -0.3, -4.5], dtype=np.float32),
                "radius": 1.0,
                "color": np.array([0.2, 0.8, 0.3], dtype=np.float32),
                "shininess": 16,
                "reflectivity": 0.06
            },
            {
                "center": np.array([-1.5, 0.4, -4.0], dtype=np.float32),
                "radius": 0.7,
                "color": np.array([0.2, 0.4, 1.0], dtype=np.float32),
                "shininess": 64,
                "reflectivity": 0.10
            },
        ]
    }

    image = render(scene, camera, width, height)

    plt.figure(figsize=(10, 7))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Step 3: Shadows + Reflections (Fixed)")
    plt.show()

    plt.imsave("step3_fixed.png", image)


if __name__ == "__main__":
    main()