import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Math helpers
# ----------------------------

def normalize(v):
    """Return a unit-length version of vector v."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# ----------------------------
# Ray-object intersection
# ----------------------------

def ray_sphere_intersect(ray_origin, ray_dir, sphere):
    """
    Returns the nearest positive t where the ray hits the sphere,
    or None if there is no hit.

    Ray:    p(t) = ray_origin + t * ray_dir
    Sphere: ||p - center||^2 = radius^2
    """
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

    # We want the closest positive hit
    eps = 1e-6
    valid_ts = [t for t in (t1, t2) if t > eps]

    if not valid_ts:
        return None

    return min(valid_ts)


# ----------------------------
# Ray tracing
# ----------------------------

def trace_ray(ray_origin, ray_dir, scene):
    """
    Find the closest object hit by the ray.
    If hit, return that object's solid color.
    Otherwise return the background color.
    """
    closest_t = float("inf")
    hit_color = scene["background"]

    for sphere in scene["spheres"]:
        t = ray_sphere_intersect(ray_origin, ray_dir, sphere)
        if t is not None and t < closest_t:
            closest_t = t
            hit_color = sphere["color"]

    return hit_color


# ----------------------------
# Rendering
# ----------------------------

def render(scene, camera, width, height):
    """
    Render the scene into an image array of shape (height, width, 3).
    """
    image = np.zeros((height, width, 3), dtype=np.float32)

    aspect_ratio = width / height
    viewport_height = 2.0
    viewport_width = viewport_height * aspect_ratio

    # Camera setup
    origin = camera["position"]

    # We place the image plane 1 unit in front of the camera
    # assuming the camera looks toward negative z.
    image_plane_z = -1.0

    for j in range(height):
        for i in range(width):
            # Convert pixel center to normalized screen coordinates in [-1, 1]
            u = (i + 0.5) / width
            v = (j + 0.5) / height

            x = (2 * u - 1) * (viewport_width / 2)
            y = (1 - 2 * v) * (viewport_height / 2)   # flip y so top row is top of image
            z = image_plane_z

            pixel_pos = np.array([x, y, z], dtype=np.float32)
            ray_dir = normalize(pixel_pos - origin)

            color = trace_ray(origin, ray_dir, scene)
            image[j, i] = color

    return image


# ----------------------------
# Main / MVP scene
# ----------------------------

def main():
    width = 800
    height = 600

    camera = {
        "position": np.array([0.0, 0.0, 0.0], dtype=np.float32)
    }

    scene = {
        "background": np.array([0.02, 0.02, 0.05], dtype=np.float32),  # dark blue-black
        "spheres": [
            {
                "center": np.array([0.0, 0.0, -3.5], dtype=np.float32),
                "radius": 0.9,
                "color": np.array([1.0, 0.2, 0.2], dtype=np.float32),   # red
            },
            {
                "center": np.array([1.4, -0.3, -4.5], dtype=np.float32),
                "radius": 1.0,
                "color": np.array([0.2, 0.8, 0.3], dtype=np.float32),   # green
            },
            {
                "center": np.array([-1.5, 0.4, -4.0], dtype=np.float32),
                "radius": 0.7,
                "color": np.array([0.2, 0.4, 1.0], dtype=np.float32),   # blue
            },
        ]
    }

    image = render(scene, camera, width, height)

    plt.figure(figsize=(10, 7))
    plt.imshow(np.clip(image, 0, 1))
    plt.axis("off")
    plt.title("Step 1 MVP: Basic Ray Tracer")
    plt.show()

    # Optional save
    plt.imsave("step1_basic_raytracer.png", np.clip(image, 0, 1))


if __name__ == "__main__":
    main()