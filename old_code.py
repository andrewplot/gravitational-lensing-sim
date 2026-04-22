import matplotlib.pyplot as plt
import numpy as np

c = 299792458
G = 6.673e-11
solar_mass = 2e30 #solar mass


class Ray:
    def __init__(self, pos, dir):
        self.pos = np.array(pos, dtype=float)
        self.dir = np.array(dir, dtype=float)
        self.path = [self.pos.copy()] #stores prev points

class BlackHole:
    def __init__(self, pos, mass):
        self.pos = np.array(pos, dtype=float)
        self.mass = mass * solar_mass
        self.radius = (2 * G * mass) / (c ** 2) #schwarzschild radius

def build_rays(n_rays, x_start, y_min, y_max):
    ray_list = []
    y_coords = np.linspace(y_min, y_max, n_rays) #evenly spaced y coords
    for y in y_coords:
        ray = Ray(pos=[x_start, y], dir=[1,0])
        ray_list.append(ray)
    return ray_list

def propagate_ray(ray, steps, dt):
    for i in range(steps):
        ray.pos += ray.dir * dt
        ray.path.append(ray.pos.copy())

def plot_rays(rays):
    plt.figure(figsize=(8,6))
    for ray in rays:
        path = np.array(ray.path)
        plt.plot(path[:,0], path[:,1], color='black')
    #plt.title("Straight 2D Rays")
    plt.show()

def main():
    n_rays = 10

    ray_list = build_rays(n_rays, x_start=0, y_min=-5, y_max=5)
    for ray in ray_list:
        propagate_ray(ray, steps=50, dt=0.5)
    
    plot_rays(ray_list)



if __name__ == "__main__":
    main()