import sys
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
from ctypes import c_void_p

from loadobj import load_objects_from_txt
from loadshader import load_shaders

WIDTH, HEIGHT = 800, 600

# Camera globals similar to C++ version
projection = None
view = None
model = None
mvp = None
eye = np.array([9.71, 4.26, 6.83], dtype=np.float32)


def update_mvp():
    global mvp
    mvp = projection @ view @ model


def rotate_matrix(axis, theta):
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return np.eye(3, dtype=np.float32)


def key_callback(window, key, scancode, action, mods):
    global eye
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    theta = np.pi / 60.0
    scale = 1.1
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_A:
        eye = rotate_matrix('y', theta) @ eye
    elif key == glfw.KEY_D:
        eye = rotate_matrix('y', -theta) @ eye
    elif key == glfw.KEY_W:
        eye = rotate_matrix('x', theta) @ eye
    elif key == glfw.KEY_S:
        eye = rotate_matrix('x', -theta) @ eye
    elif key == glfw.KEY_E:
        eye = eye * scale
    elif key == glfw.KEY_Q and np.linalg.norm(eye) > 1.0:
        eye = eye / scale
    update_mvp()


def init_window():
    if not glfw.init():
        sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WIDTH, HEIGHT, 'Python Renderer', None, None)
    if not window:
        glfw.terminate()
        sys.exit(1)
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    return window


def create_vbo(objs):
    vertices = np.concatenate([np.array(obj.vertices, dtype=np.float32) for obj in objs])
    uvs = np.concatenate([np.array(obj.uvs, dtype=np.float32) for obj in objs])
    normals = np.concatenate([np.array(obj.normals, dtype=np.float32) for obj in objs])
    count = len(vertices)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo_vertices = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(0)

    vbo_uvs = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_uvs)
    glBufferData(GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(1)

    vbo_normals = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(2)

    return vao, count


def main():
    global projection, view, model, mvp
    window = init_window()

    objs = load_objects_from_txt('scene.txt')
    vao, count = create_vbo(objs)

    shader = load_shaders([
        'shader/vertex.glsl',
        'shader/fragment.glsl'
    ], [GL_VERTEX_SHADER, GL_FRAGMENT_SHADER])

    projection = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, -2/(100-0.1), -(100+0.1)/(100-0.1)],
         [0, 0, 0, 1]], dtype=np.float32)
    view = np.identity(4, dtype=np.float32)
    model = np.identity(4, dtype=np.float32)
    update_mvp()

    mvp_loc = glGetUniformLocation(shader, 'MVP')
    eye_loc = glGetUniformLocation(shader, 'Eye')

    glUseProgram(shader)
    while not glfw.window_should_close(window):
        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp)
        glUniform3fv(eye_loc, 1, eye)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, count)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == '__main__':
    main()
