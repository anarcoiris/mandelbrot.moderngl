#!/usr/bin/env python3
"""
fractal_glsl_moderngl_v3.py

Versión 3: añade export por tiles (alto detalle) y rendering background con create_standalone_context cuando sea posible.
Características añadidas:
 - Exportar imagen de alta resolución por tiles (tecla E). Por defecto guarda 3840x2160.
 - Supersampling por tile (ajustable) para reducir aliasing.
 - Uso de moderngl.create_standalone_context() en hilo background para renderizar tiles sin bloquear la UI (si fallase, hace render en el hilo principal).
 - Información de progreso impresa en consola y mostrada en la esquina de la ventana.
 - Conserva controles interactivos (zoom con rueda, pan con botón derecho, grabar puntos con clic izquierdo).

Dependencias: pip install moderngl pyglet Pillow numpy

Limitaciones/Notas:
 - El export puede ser lento según resolución y iteraciones; el tile size predeterminado es 1024 (ajustable). Comprueba ctx.max_texture_size si tienes errores.
 - Precision: GLSL float32 limita la profundidad de zoom útil.

Controles en runtime:
 - Rueda: zoom (centrado en cursor)
 - Botón derecho + drag: pan
 - Izquierdo click: grabar punto
 - SPACE: alternar z0 (c / 0)
 - [: reducir iteraciones
 - ]: aumentar iteraciones
 - S: guardar screenshot de la ventana
 - E: Exportar alta resolución por tiles (usa constantes por defecto, o modifica el script)
 - R: reset
 - C: limpiar puntos

"""

import sys
import time
import math
import threading
import numpy as np
from PIL import Image
import moderngl
import pyglet
from pyglet.window import key, mouse

# -----------------------
# Config (ajusta según tus necesidades / GPU)
# -----------------------
WINDOW_SIZE = (1000, 700)
INIT_CENTER = (-0.5, 0.0)
INIT_SCALE = 2.0
INIT_ITER = 400
ESCAPE_RADIUS = 4.0

# Export defaults (tecla E)
EXPORT_WIDTH = 3840
EXPORT_HEIGHT = 2160
EXPORT_TILE = 1024  # tile size in pixels (prefer power of two, <= ctx.max_texture_size)
EXPORT_SUPERSAMPLE = 1  # 1 = none, 2 = 2x supersample, etc.
EXPORT_ITER = None  # if None, use current iter_max

# Shader sources (vertex + fragment)
VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_pos;
void main() {
    v_pos = in_pos;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 v_pos;
out vec4 f_color;

uniform vec2 u_center;
uniform float u_scale;
uniform ivec2 u_resolution;
uniform int u_iter_max;
uniform float u_escape;
uniform int u_z0_mode;

vec3 hsv2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x*6.0 + vec3(0.0,4.0,2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    rgb = rgb * rgb * (3.0 - 2.0 * rgb);
    return c.z * mix(vec3(1.0), rgb, c.y);
}

void main() {
    float aspect = float(u_resolution.x) / float(u_resolution.y);
    vec2 uv = v_pos;
    vec2 c = vec2(u_center.x + uv.x * u_scale * aspect, u_center.y + uv.y * u_scale);

    float zr = (u_z0_mode == 0) ? c.x : 0.0;
    float zi = (u_z0_mode == 0) ? c.y : 0.0;
    float cr = c.x;
    float ci = c.y;
    float escape2 = u_escape * u_escape;

    float smoothIter = float(u_iter_max);
    for (int i = 0; i < 2000; ++i) {
        if (i >= u_iter_max) { break; }
        float zr2 = zr*zr;
        float zi2 = zi*zi;
        if (zr2 + zi2 > escape2) {
            float modz = sqrt(zr2 + zi2);
            smoothIter = float(i) + 1.0 - log(log(modz))/log(2.0);
            break;
        }
        float newzr = zr2 - zi2 + cr;
        float newzi = 2.0*zr*zi + ci;
        zr = newzr;
        zi = newzi;
    }

    float t = smoothIter / float(u_iter_max);
    float hue = 0.95 - 0.85 * t;
    float sat = 0.9;
    float val = 0.6 + 0.4*(1.0 - t);
    vec3 col = hsv2rgb(vec3(hue, sat, val));
    if (smoothIter >= float(u_iter_max)) { col = vec3(0.0); }
    f_color = vec4(col, 1.0);
}
"""

# -----------------------
# Helper: create program + vao in a given context
# -----------------------

def setup_program_and_vao(ctx):
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
        -1.0,  1.0,
         1.0, -1.0,
         1.0,  1.0,
    ], dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '2f', 'in_pos')])
    return prog, vao

# -----------------------
# Main Window Application
# -----------------------
class FractalWindow(pyglet.window.Window):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # moderngl context from window
        self.ctx = moderngl.create_context()
        self.ctx.viewport = (0, 0, self.width, self.height)
        # print limits
        try:
            print('GL max texture size:', self.ctx.max_texture_size)
        except Exception:
            pass

        self.prog, self.vao = setup_program_and_vao(self.ctx)

        # state
        self.center = list(INIT_CENTER)
        self.view_scale = INIT_SCALE
        self.iter_max = INIT_ITER
        self.escape = ESCAPE_RADIUS
        self.z0_mode = 0
        self.points = []

        # UI
        self.info_label = pyglet.text.Label('', x=10, y=self.height-22)
        self.hint = pyglet.text.Label("SPACE z0 | [ ] iter | E export | S save | R reset | C clear points",
                                     x=10, y=8)

        # mouse
        self._dragging = False
        self._last_mouse = None

        # export state
        self.export_thread = None
        self.export_busy = False
        self.export_progress = (0, 0)

        # initial render
        self.render()

    # safe uniform setter (ignore missing uniforms)
    def set_uniform_safe(self, name, value):
        try:
            self.prog[name].value = value
        except Exception:
            try:
                # for ivec2 style
                self.prog[name].value = tuple(value)
            except Exception:
                pass

    def update_uniforms(self, prog=None):
        if prog is None:
            prog = self.prog
        def s(name, val):
            try:
                prog[name].value = val
            except Exception:
                pass
        s('u_center', tuple(self.center))
        s('u_scale', float(self.view_scale))
        s('u_iter_max', int(self.iter_max))
        s('u_escape', float(self.escape))
        s('u_z0_mode', int(self.z0_mode))
        s('u_resolution', (self.width, self.height))

    def on_draw(self):
        self.render()

    def render(self):
        self.ctx.clear(0.0, 0.0, 0.0)
        self.update_uniforms()
        self.vao.render(moderngl.TRIANGLES)
        # labels
        self.info_label.text = f"center=({self.center[0]:.6g},{self.center[1]:.6g}) scale={self.view_scale:.6g} iter={self.iter_max} z0={'c' if self.z0_mode==0 else '0'} pts={len(self.points)}"
        self.info_label.draw()
        self.hint.draw()
        if self.export_busy:
            # draw progress text
            done, total = self.export_progress
            prog_txt = f'Exporting tiles: {done}/{total}'
            pyglet.text.Label(prog_txt, x=10, y=40, color=(255,255,0,255)).draw()

    def screen_to_complex_for_size(self, px, py, full_w, full_h, center, view_scale):
        # px,py in [0..full_w-1], origin bottom-left
        nx = (px / full_w) * 2.0 - 1.0
        ny = (py / full_h) * 2.0 - 1.0
        aspect = float(full_w) / float(full_h)
        re = center[0] + nx * view_scale * aspect
        im = center[1] + ny * view_scale
        return re, im

    def map_tile_to_center(self, tile_x, tile_y, tile_w, tile_h, full_w, full_h):
        # tile_x,y are pixel coordinates of tile origin (left,bottom)
        cx = tile_x + tile_w / 2.0
        cy = tile_y + tile_h / 2.0
        return self.screen_to_complex_for_size(cx, cy, full_w, full_h, self.center, self.view_scale)

    def export_highres_tiles(self, full_w, full_h, tile_size=EXPORT_TILE, supersample=EXPORT_SUPERSAMPLE, iter_override=None, out_fname=None):
        """
        Render a full_w x full_h image by rendering tiles to an offscreen standalone context.
        """
        if self.export_busy:
            print('Export already running')
            return
        self.export_busy = True
        self.export_progress = (0, 1)

        def worker():
            nonlocal full_w, full_h, tile_size, supersample, iter_override, out_fname
            print('Starting export:', full_w, 'x', full_h, 'tile_size=', tile_size, 'supersample=', supersample)
            # create standalone context for background rendering if possible
            try:
                sctx = moderngl.create_standalone_context()
                print('Standalone context created for export')
                use_standalone = True
            except Exception as e:
                print('Could not create standalone context, will render in main context (may block UI). Error:', e)
                sctx = self.ctx
                use_standalone = False

            # check max texture size
            try:
                max_tex = sctx.max_texture_size
                if tile_size * supersample > max_tex:
                    print(f'Warning: requested tile_size*supersample > max_texture_size ({max_tex}). Reducing tile size.')
                    tile_size_eff = max_tex
                else:
                    tile_size_eff = tile_size * supersample
            except Exception:
                tile_size_eff = tile_size * supersample

            # prepare program/vao in sctx
            prog, vao = setup_program_and_vao(sctx)

            # output canvas
            canvas = Image.new('RGB', (full_w, full_h))

            cols = math.ceil(full_w / tile_size)
            rows = math.ceil(full_h / tile_size)
            total_tiles = cols * rows
            done_tiles = 0
            target_iter = iter_override if iter_override is not None else self.iter_max

            # iterate tiles row-major bottom-to-top
            for row in range(rows):
                for col in range(cols):
                    tx = col * tile_size
                    ty = row * tile_size
                    tw = min(tile_size, full_w - tx)
                    th = min(tile_size, full_h - ty)
                    # render target size (apply supersample)
                    render_w = int(tw * supersample)
                    render_h = int(th * supersample)

                    # compute tile center in complex plane
                    tile_center = self.map_tile_to_center(tx, ty, tw, th, full_w, full_h)

                    # create fbo and render
                    try:
                        fbo = sctx.simple_framebuffer((render_w, render_h))
                        fbo.use()
                        sctx.viewport = (0,0,render_w,render_h)
                        # set uniforms on prog (safe)
                        def sset(name, val):
                            try:
                                prog[name].value = val
                            except Exception:
                                pass
                        sset('u_center', tuple(tile_center))
                        sset('u_scale', float(self.view_scale))
                        sset('u_iter_max', int(target_iter))
                        sset('u_escape', float(self.escape))
                        sset('u_z0_mode', int(self.z0_mode))
                        sset('u_resolution', (render_w, render_h))

                        vao.render(moderngl.TRIANGLES)
                        data = fbo.read(components=3, alignment=1)
                        img = Image.frombytes('RGB', (render_w, render_h), data).transpose(Image.FLIP_TOP_BOTTOM)
                        # downsample if supersample>1
                        if supersample > 1:
                            img = img.resize((tw, th), resample=Image.LANCZOS)
                        else:
                            if (render_w, render_h) != (tw, th):
                                img = img.crop((0, 0, tw, th))
                        # paste into canvas (note PIL origin top-left, our ty is bottom)
                        paste_y = full_h - (ty + th)  # convert to top-left origin
                        canvas.paste(img, (tx, paste_y))

                        done_tiles += 1
                        self.export_progress = (done_tiles, total_tiles)
                        print(f'[{done_tiles}/{total_tiles}] tile ({col},{row}) -> center {tile_center} iter={target_iter} size={tw}x{th}')

                    except Exception as e:
                        print('Error rendering tile:', e)
                        # continue with next tile
                        continue

            # save output
            if out_fname is None:
                out_fname = f'fractal_export_{full_w}x{full_h}_{int(time.time())}.png'
            canvas.save(out_fname)
            print('Export finished:', out_fname)

            # cleanup
            if use_standalone:
                try:
                    sctx.release()
                except Exception:
                    pass
            self.export_busy = False
            self.export_progress = (0, 0)

        # launch worker thread
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self.export_thread = t

    # user input / events
    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            re, im = self.screen_to_complex_for_size(x, y, self.width, self.height, self.center, self.view_scale)
            self.points.append((re, im))
            print(f'Recorded point {len(self.points)-1}: {re} + {im}i')
        elif button == mouse.RIGHT:
            self._dragging = True
            self._last_mouse = (x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.RIGHT:
            self._dragging = False
            self._last_mouse = None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.RIGHT:
            aspect = self.width / self.height
            dxc = -dx / self.width * 2.0 * self.view_scale * aspect
            dyc = -dy / self.height * 2.0 * self.view_scale
            self.center[0] += dxc
            self.center[1] += dyc
            self.render()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        factor = 1.0 / (1.15 ** scroll_y)
        nx = (x / self.width) * 2.0 - 1.0
        ny = (y / self.height) * 2.0 - 1.0
        aspect = self.width / self.height
        old_re = self.center[0] + nx * self.view_scale * aspect
        old_im = self.center[1] + ny * self.view_scale
        self.view_scale *= factor
        self.center[0] = old_re - nx * self.view_scale * aspect
        self.center[1] = old_im - ny * self.view_scale
        self.render()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.z0_mode = 1 - self.z0_mode
            print('z0 mode ->', 'c' if self.z0_mode==0 else '0')
            self.render()
        elif symbol == key.S:
            self.save_screenshot()
        elif symbol == key.R:
            self.center = list(INIT_CENTER)
            self.view_scale = INIT_SCALE
            self.render()
        elif symbol == key.C:
            self.points = []
            print('Points cleared')
            self.render()
        elif symbol == key.LEFTBRACKET:
            self.iter_max = max(8, int(self.iter_max * 0.8))
            print('iter ->', self.iter_max)
            self.render()
        elif symbol == key.RIGHTBRACKET:
            self.iter_max = int(self.iter_max * 1.25)
            print('iter ->', self.iter_max)
            self.render()
        elif symbol == key.E:
            # start export with defaults
            target_iter = EXPORT_ITER if EXPORT_ITER is not None else self.iter_max
            print(f'Starting export {EXPORT_WIDTH}x{EXPORT_HEIGHT} tilesize={EXPORT_TILE} ss={EXPORT_SUPERSAMPLE} iter={target_iter}')
            self.export_highres_tiles(EXPORT_WIDTH, EXPORT_HEIGHT, tile_size=EXPORT_TILE, supersample=EXPORT_SUPERSAMPLE, iter_override=target_iter)

    def save_screenshot(self):
        # render current framebuffer to image and save
        fbo = self.ctx.simple_framebuffer((self.width, self.height))
        fbo.use()
        self.vao.render(moderngl.TRIANGLES)
        data = fbo.read(components=3, alignment=1)
        img = Image.frombytes('RGB', (self.width, self.height), data).transpose(Image.FLIP_TOP_BOTTOM)
        fname = f'fractal_screenshot_{int(time.time())}.png'
        img.save(fname)
        print('Saved', fname)
        self.ctx.screen.use()

# -----------------------
# run
# -----------------------

def main():
    try:
        config = pyglet.gl.Config(double_buffer=True, major_version=3, minor_version=3)
        win = FractalWindow(width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], caption='Fractal GLSL v3', resizable=True, config=config)
    except Exception as e:
        print('GL3 config failed, trying default context:', e)
        win = FractalWindow(width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], caption='Fractal GLSL v3', resizable=True)

    pyglet.app.run()

if __name__ == '__main__':
    main()
