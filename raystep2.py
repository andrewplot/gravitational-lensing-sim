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
# Ray-sphere intersection
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

    if not valid_ts:
        return None

    return min(valid_ts)


# ----------------------------
# Phong shading
# ----------------------------

def phong_shading(point, normal, view_dir, sphere, light):
    light_dir = normalize(light["position"] - point)

    # --- Ambient ---
    ambient = light["ambient"] * sphere["color"]

    # --- Diffuse ---
    diff = max(np.dot(normal, light_dir), 0.0)
    diffuse = light["diffuse"] * diff * sphere["color"]

    # --- Specular ---
    reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
    spec = max(np.dot(view_dir, reflect_dir), 0.0) ** sphere["shininess"]
    specular = light["specular"] * spec * light["color"]

    return ambient + diffuse + specular


# ----------------------------
# Ray tracing
# ----------------------------

def trace_ray(ray_origin, ray_dir, scene, camera):
    closest_t = float("inf")
    hit_sphere = None

    for sphere in scene["spheres"]:
        t = ray_sphere_intersect(ray_origin, ray_dir, sphere)
        if t is not None and t < closest_t:
            closest_t = t
            hit_sphere = sphere

    if hit_sphere is None:
        return scene["background"]

    # --- Compute hit point ---
    hit_point = ray_origin + closest_t * ray_dir

    # --- Compute normal ---
    normal = normalize(hit_point - hit_sphere["center"])

    # --- View direction ---
    view_dir = normalize(camera["position"] - hit_point)

    # --- Lighting ---
    color = phong_shading(hit_point, normal, view_dir, hit_sphere, scene["light"])

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

            color = trace_ray(origin, ray_dir, scene, camera)
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
        "background": np.array([0.02, 0.02, 0.05]),

        "light": {
            "position": np.array([5.0, 5.0, 0.0]),
            "color": np.array([1.0, 1.0, 1.0]),
            "ambient": 0.1,
            "diffuse": 0.7,
            "specular": 0.5
        },

        "spheres": [
            {
                "center": np.array([0.0, 0.0, -3.5]),
                "radius": 0.9,
                "color": np.array([1.0, 0.2, 0.2]),
                "shininess": 32
            },
            {
                "center": np.array([1.4, -0.3, -4.5]),
                "radius": 1.0,
                "color": np.array([0.2, 0.8, 0.3]),
                "shininess": 16
            },
            {
                "center": np.array([-1.5, 0.4, -4.0]),
                "radius": 0.7,
                "color": np.array([0.2, 0.4, 1.0]),
                "shininess": 64
            },
        ]
    }

    image = render(scene, camera, width, height)

    plt.imshow(image)
    plt.axis("off")
    plt.title("Step 2: Phong Shading")
    plt.show()

    plt.imsave("step2_phong.png", image)


if __name__ == "__main__":
    main()