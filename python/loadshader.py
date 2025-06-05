from OpenGL.GL import *


def load_shaders(shader_paths, stages):
    program = glCreateProgram()
    shader_ids = []
    for path, stage in zip(shader_paths, stages):
        with open(path, 'r') as f:
            source = f.read()
        shader = glCreateShader(stage)
        glShaderSource(shader, source)
        glCompileShader(shader)
        result = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not result:
            print(glGetShaderInfoLog(shader).decode())
        glAttachShader(program, shader)
        shader_ids.append(shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        print(glGetProgramInfoLog(program).decode())
    for shader in shader_ids:
        glDetachShader(program, shader)
        glDeleteShader(shader)
    return program
