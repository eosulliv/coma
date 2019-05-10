import numpy as np

from mayavi import mlab
from pynput import keyboard

gold = (244 / 256, 178 / 256, 88 / 256)
red = (144 / 256, 12 / 256, 63 / 256)


def visualize_latent_space(model, facedata, mesh_path=None, bgcolor=(1, 1, 1), size=(600, 600)):
    if mesh_path is not None:
        normalized_mesh = facedata.get_normalized_meshes([mesh_path])
    else:
        normalized_mesh = np.array([facedata.vertices_test[0]])

    fig = mlab.figure(bgcolor=bgcolor, size=size)
    fig.scene.z_plus_view()

    latent_vector = model.encode(normalized_mesh)
    recon_vec = model.decode(latent_vector)
    mesh_mesh = facedata.vec2mesh(recon_vec)

    ms = mlab.triangular_mesh(mesh_mesh.points[:, 0], mesh_mesh.points[:, 1],
                              mesh_mesh.points[:, 2], mesh_mesh.trilist, color=gold)

    def on_press(key):
        try:
            if key.char == "q":
                latent_vector[0][0] = latent_vector[0][0] + 0.1
            elif key.char == "w":
                latent_vector[0][1] = latent_vector[0][1] + 0.1
            elif key.char == "e":
                latent_vector[0][2] = latent_vector[0][2] + 0.1
            elif key.char == "r":
                latent_vector[0][3] = latent_vector[0][3] + 0.1
            elif key.char == "t":
                latent_vector[0][4] = latent_vector[0][4] + 0.1
            elif key.char == "y":
                latent_vector[0][5] = latent_vector[0][5] + 0.1
            elif key.char == "u":
                latent_vector[0][6] = latent_vector[0][6] + 0.1
            elif key.char == "i":
                latent_vector[0][7] = latent_vector[0][7] + 0.1

            elif key.char == "a":
                latent_vector[0][0] = latent_vector[0][0] - 0.1
            elif key.char == "s":
                latent_vector[0][1] = latent_vector[0][1] - 0.1
            elif key.char == "d":
                latent_vector[0][2] = latent_vector[0][2] - 0.1
            elif key.char == "f":
                latent_vector[0][3] = latent_vector[0][3] - 0.1
            elif key.char == "g":
                latent_vector[0][4] = latent_vector[0][4] - 0.1
            elif key.char == "h":
                latent_vector[0][5] = latent_vector[0][5] - 0.1
            elif key.char == "j":
                latent_vector[0][6] = latent_vector[0][6] - 0.1
            elif key.char == "k":
                latent_vector[0][7] = latent_vector[0][7] - 0.1
            elif key.char == "\x1b":
                listener.stop()
                print('Exiting...')

            print('Key {} pressed. Latent vector: {}'.format(key.char, latent_vector))

            rec_vec = model.decode(latent_vector)
            mesh = facedata.vec2mesh(rec_vec)
            ms.mlab_source.points = mesh.points
        except AttributeError:
            listener.stop()
            print('Exiting...')

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    @mlab.animate
    def anim():
        while listener.running:
            yield

    anim()
    mlab.show()
