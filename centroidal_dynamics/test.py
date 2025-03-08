import meshcat
viz = meshcat.Visualizer()
viz.open()
viz.set_object(meshcat.geometry.Sphere(0.5))