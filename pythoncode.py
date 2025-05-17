import OpenGL.GL as gl
import glfw
import numpy as np
import ctypes
import os
from PIL import Image
import struct
import zipfile
import trimesh

# ==== Shader Loader ====
def read_file(path):
    with open(path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
        return f.read()

VERTEX_SHADER   = read_file("vertex.glsl")
FRAGMENT_SHADER = read_file("fragment.glsl")
COMPUTE_SHADER  = read_file("shader.comp")

# ==== Triangle Data ====
triangle_dtype = np.dtype({
    'names': [
      'v0',
      'v1',
      'v2',
      'material',
      'color',
      'emission',
      'emissionStrength',
    ],
    'formats': [
      '3f4',   # v0 : vec3
      '3f4',   # v1 : vec3
      '3f4',   # v2 : vec3
      'i4',    # material : int
      '3f4',   # color : vec3
      '3f4',   # emission : vec3
      'f4',    # emissionStrength : float
    ],
    'offsets': [
      0,   # v0
      16,  # v1
      32,  # v2
      44,  # material
      48,  # color
      64,  # emission
      76,  # emissionStrength
    ],
    'itemsize': 80
})

# ==== Load Knight Model ====
mesh = trimesh.load('knight.glb', force='mesh')
verts = mesh.vertices       # (V,3) array of vertex positions
faces = mesh.faces          # (F,3) array of triangle indices

# Pack into structured array
triangles = np.zeros(len(faces), dtype=triangle_dtype)
for i, (i0, i1, i2) in enumerate(faces):
    triangles['v0'][i] = verts[i0]
    triangles['v1'][i] = verts[i1]
    triangles['v2'][i] = verts[i2]
    triangles['material'][i]         = 0
    triangles['color'][i]            = (1.0,1.0,1.0)
    triangles['emission'][i]         = (0.0,0.0,0.0)
    triangles['emissionStrength'][i] = 0.0

# Center + uniformly scale to fit inside 2×2×2 Cornell box
all_verts = np.vstack([triangles['v0'],
                       triangles['v1'],
                       triangles['v2']])
min_c = all_verts.min(axis=0)
max_c = all_verts.max(axis=0)
center = (min_c + max_c) * 0.5
max_extent = (max_c - min_c).max()
scale = 1.0 / max_extent  # scale to fit inside 2x2x2 box

theta = np.radians(45.0)
c, s = np.cos(theta), np.sin(theta)
R = np.array([
    [ c, 0,  s],
    [ 0, 1,  0],
    [-s, 0,  c]
], dtype=np.float32)
pivot = center.astype(np.float32)

for fld in ('v0','v1','v2'):
    triangles[fld] = (triangles[fld] - center) * scale # scale
    triangles[fld] = (triangles[fld] - pivot) @ R.T + pivot # rotate

#adjust the location of the triangles
triangles['v0'] += np.array([0.0, -0.5, -0.25])
triangles['v1'] += np.array([0.0, -0.5, -0.25])
triangles['v2'] += np.array([0.0, -0.5, -0.25])

# gather all vertices
all_verts = np.vstack([
    triangles['v0'],
    triangles['v1'],
    triangles['v2'],
])

# for the bounding box
bb_min = all_verts.min(axis=0).astype(np.float32)
bb_max = all_verts.max(axis=0).astype(np.float32)

# ==== Screenshot Utility ====
def save_image(filename, width, height):
    # Bind default framebuffer and read from the front buffer after swap
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    gl.glReadBuffer(gl.GL_FRONT)
    data = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
    image = np.flipud(image)
    image = np.ascontiguousarray(image)
    img = Image.fromarray(image, 'RGBA')
    img.save(filename)
    print(f"Screenshot saved to {filename} ({width}x{height})")

# ==== Input Callback ====
def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_K and action == glfw.PRESS:
        w, h = glfw.get_framebuffer_size(window)
        save_image("screenshot.png", w, h)

# ==== Shader Compilation ====
def compile_shader(source, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        raise RuntimeError(gl.glGetShaderInfoLog(shader).decode())
    return shader

# ==== Program Creation ====
def create_program(vertex_src, fragment_src):
    program = gl.glCreateProgram()
    vs = compile_shader(vertex_src, gl.GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, gl.GL_FRAGMENT_SHADER)
    gl.glAttachShader(program, vs)
    gl.glAttachShader(program, fs)
    gl.glLinkProgram(program)
    return program

def create_compute_program(source):
    program = gl.glCreateProgram()
    cs = compile_shader(source, gl.GL_COMPUTE_SHADER)
    gl.glAttachShader(program, cs)
    gl.glLinkProgram(program)
    return program

# ==== Fullscreen Quad ====
def draw_fullscreen_quad():
    quad = np.array([
        -1, -1, 0, 0,
         1, -1, 1, 0,
        -1,  1, 0, 1,
         1,  1, 1, 1,
    ], dtype=np.float32)

    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)

    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(8))

    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

    gl.glDeleteBuffers(1, [vbo])
    gl.glDeleteVertexArrays(1, [vao])

# ==== Main Application ====
def main():
    MAX_ACCUM_FRAMES = 500
    INIT_W, INIT_H = 500, 500

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(INIT_W, INIT_H, "Real-Time Raytracer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create window")

    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)

    triangle_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, triangle_buffer)
    gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, triangles.nbytes, triangles, gl.GL_STATIC_DRAW)
    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 4, triangle_buffer)

    # Compile shaders
    compute_prog = create_compute_program(COMPUTE_SHADER)
    render_prog  = create_program(VERTEX_SHADER, FRAGMENT_SHADER)

    # Determine actual framebuffer size (handles DPI scaling)
    WIDTH, HEIGHT = glfw.get_framebuffer_size(window)

    # Create textures for output and accumulation
    tex       = gl.glGenTextures(1)
    accum_tex = gl.glGenTextures(1)
    for t in (tex, accum_tex):
        gl.glBindTexture(gl.GL_TEXTURE_2D, t)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, WIDTH, HEIGHT, 0,
                        gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

    # Timing and frame counters
    fps_counter = 0
    frame_count = 0
    last_time   = glfw.get_time()

    # Camera state
    camera_pos   = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    camera_speed = 0.025
    yaw, pitch   = -0.02 , -0.1
    prev_cam     = {"pos": camera_pos.copy(), "yaw": yaw, "pitch": pitch}

    while not glfw.window_should_close(window):
        t = glfw.get_time()
        print(camera_pos, yaw, pitch)

        # Build camera basis
        front = np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            -np.cos(pitch) * np.cos(yaw)
        ], dtype=np.float32)
        front /= np.linalg.norm(front)
        up = np.cross(np.cross([0,1,0], front), front)
        up /= np.linalg.norm(up)
        right = np.cross(front, up)

        # Handle movement
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS: camera_pos += camera_speed * front
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS: camera_pos -= camera_speed * front
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: camera_pos -= camera_speed * right
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: camera_pos += camera_speed * right
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS: camera_pos += camera_speed * up
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS: camera_pos -= camera_speed * up
        if glfw.get_key(window, glfw.KEY_RIGHT)  == glfw.PRESS: yaw   -= 0.02
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS: yaw   += 0.02
        if glfw.get_key(window, glfw.KEY_UP)    == glfw.PRESS: pitch += 0.02
        if glfw.get_key(window, glfw.KEY_DOWN)  == glfw.PRESS: pitch -= 0.02
        pitch = np.clip(pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)

        # Update random seed
        seed = np.random.rand()

        # Handle resize: reallocate textures if framebuffer size changed
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        if fb_w != WIDTH or fb_h != HEIGHT:
            WIDTH, HEIGHT = fb_w, fb_h
            frame_count = 0
            for t in (tex, accum_tex):
                gl.glBindTexture(gl.GL_TEXTURE_2D, t)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, WIDTH, HEIGHT, 0,
                                gl.GL_RGBA, gl.GL_FLOAT, None)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # Dispatch compute shader for accumulation
        if frame_count < MAX_ACCUM_FRAMES:

            gl.glUseProgram(compute_prog)
            loc_min = gl.glGetUniformLocation(compute_prog, "meshMin")
            loc_max = gl.glGetUniformLocation(compute_prog, "meshMax")
            gl.glUniform3fv(loc_min, 1, bb_min)
            gl.glUniform3fv(loc_max, 1, bb_max)
            gl.glUniform3fv(gl.glGetUniformLocation(compute_prog, "cameraPos"),     1, camera_pos)
            gl.glUniform3fv(gl.glGetUniformLocation(compute_prog, "cameraForward"), 1, front)
            gl.glUniform3fv(gl.glGetUniformLocation(compute_prog, "cameraRight"),   1, right)
            gl.glUniform3fv(gl.glGetUniformLocation(compute_prog, "cameraUp"),      1, up)
            gl.glUniform1f(gl.glGetUniformLocation(compute_prog, "iTime"), t)
            gl.glUniform1i(gl.glGetUniformLocation(compute_prog, "frameCount"), frame_count)
            gl.glUniform1i(gl.glGetUniformLocation(compute_prog, "NUM_TRIANGLES"), len(triangles))
            gl.glUniform1f(gl.glGetUniformLocation(compute_prog, "randomSeed"), seed)

            gl.glBindImageTexture(0, tex,       0, False, 0, gl.GL_WRITE_ONLY, gl.GL_RGBA32F)
            gl.glBindImageTexture(1, accum_tex, 0, False, 0, gl.GL_READ_ONLY,  gl.GL_RGBA32F)
            gl.glDispatchCompute(WIDTH // 16, HEIGHT // 16, 1)
            gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

            # Copy into accumulation texture
            gl.glCopyImageSubData(
                tex,       gl.GL_TEXTURE_2D, 0, 0, 0, 0,
                accum_tex, gl.GL_TEXTURE_2D, 0, 0, 0, 0,
                WIDTH, HEIGHT, 1
            )

        # Render the full-screen quad
        gl.glUseProgram(render_prog)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        draw_fullscreen_quad()

        glfw.swap_buffers(window)
        glfw.poll_events()

        # Reset accumulation on camera move
        moved = (not np.allclose(camera_pos, prev_cam["pos"], atol=1e-6) or
                 abs(yaw   - prev_cam["yaw"])   > 1e-6 or
                 abs(pitch - prev_cam["pitch"]) > 1e-6)
        if moved:
            frame_count = 0
            prev_cam["pos"]   = camera_pos.copy()
            prev_cam["yaw"]   = yaw
            prev_cam["pitch"] = pitch
        else:
            frame_count += 1

        # Update FPS title every second
        now = glfw.get_time()
        fps_counter += 1
        if now - last_time >= 1.0:
            fps = fps_counter / (now - last_time)
            glfw.set_window_title(window, f"Real-Time Raytracer - {fps:.1f} FPS, Frame {frame_count}")
            fps_counter = 0
            last_time   = now

    glfw.terminate()

if __name__ == "__main__":
    main()