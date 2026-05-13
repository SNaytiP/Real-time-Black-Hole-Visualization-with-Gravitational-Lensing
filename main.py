import moderngl
import moderngl_window as mglw
import numpy as np
import rebound
import reboundx
from pathlib import Path
from starfield import generate_starfield
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import io

class BlackHoleApp(mglw.WindowConfig):
    title          = "Black Hole — Gravitational Lensing"
    window_size    = (1280, 720)
    aspect_ratio   = 1280 / 720
    resource_dir   = Path(__file__).parent

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ── Camera state ──────────────────────────────────────────
        self.cam_azimuth   = 45.0
        self.cam_elevation = 25.0
        self.cam_distance  = 5.0
        self.cam_target    = np.array([0.0, 0.0, 0.0])

        # ── Mouse state ───────────────────────────────────────────
        self._mouse_down   = False
        self._last_mouse   = (0, 0)

        # ── Spawned particle tracking ─────────────────────────────
        self._n_spawned = 0
        self._show_hud  = True
        self._spawn_r   = 0.8

        self._setup_simulation()
        self._setup_starfield()
        self._setup_shaders()
        self._setup_framebuffer()
        self._setup_particles()
        self._setup_hud()
        self.frame = 0

    # ── REBOUND simulation ────────────────────────────────────
    def _setup_simulation(self):
        self.sim = rebound.Simulation()
        self.sim.units = ('AU', 'yr', 'Msun')
        self.sim.add(m=10.0)

        for i in range(5):
            self.sim.add(
                m     = 1e-6,
                a     = 0.4 + i * 0.4,
                e     = np.random.uniform(0.05, 0.35),
                inc   = np.random.uniform(0, np.pi / 3),
                Omega = np.random.uniform(0, 2 * np.pi),
            )

        self.sim.move_to_com()

        # เพิ่ม GR
        self.rebx = reboundx.Extras(self.sim)
        gr = self.rebx.load_force("gr_full")
        self.rebx.add_force(gr)
        gr.params["c"] = 63241.0

    # ── Starfield texture ─────────────────────────────────────
    def _setup_starfield(self):
        img  = generate_starfield(1280, 720)
        data = img.tobytes()
        self.star_tex = self.ctx.texture((1280, 720), 3, data)
        self.star_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.star_tex.repeat_x = True
        self.star_tex.repeat_y = True

    # ── Shaders ───────────────────────────────────────────────
    def _setup_shaders(self):
        vert = Path("shaders/lensing.vert").read_text()
        frag = Path("shaders/lensing.frag").read_text()
        self.lensing_prog = self.ctx.program(
            vertex_shader   = vert,
            fragment_shader = frag,
        )

        # Full-screen quad
        vertices = np.array([
            -1, -1,  0, 0,
             1, -1,  1, 0,
            -1,  1,  0, 1,
             1,  1,  1, 1,
        ], dtype=np.float32)
        vbo = self.ctx.buffer(vertices)
        self.quad_vao = self.ctx.vertex_array(
            self.lensing_prog,
            [(vbo, '2f 2f', 'in_position', 'in_texcoord')]
        )

        # Scene shader (วาด particles)
        scene_vert = """
        #version 330
        in vec3 in_pos;
        uniform mat4 mvp;
        uniform float point_size;
        void main() { gl_Position = mvp * vec4(in_pos, 1.0);
                       gl_PointSize = point_size; }
        """
        scene_frag = """
        #version 330
        uniform vec4 color;
        out vec4 fragColor;
        void main() { fragColor = color; }
        """
        self.scene_prog = self.ctx.program(
            vertex_shader   = scene_vert,
            fragment_shader = scene_frag,
        )

        # Trail shader — fading line strip per star
        trail_vert = """
        #version 330
        in vec3 in_pos;
        in float in_age;
        uniform mat4 mvp;
        out float v_age;
        void main() {
            gl_Position = mvp * vec4(in_pos, 1.0);
            v_age = in_age;
        }
        """
        trail_frag = """
        #version 330
        uniform vec4 color;
        in float v_age;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color.rgb, color.a * v_age * v_age);
        }
        """
        self.trail_prog = self.ctx.program(
            vertex_shader   = trail_vert,
            fragment_shader = trail_frag,
        )

    # ── Framebuffer ───────────────────────────────────────────
    def _setup_framebuffer(self):
        self.scene_tex = self.ctx.texture((1280, 720), 4)
        self.scene_fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_tex]
        )

    # ── Particle buffers (accretion disk) ─────────────────────
    def _setup_particles(self):
        self.N_DISK   = 4000
        self.r        = np.random.uniform(0.15, 0.6, self.N_DISK)
        self.theta    = np.random.uniform(0, 2*np.pi, self.N_DISK)
        self.tilt     = 0.12
        self.GM       = 4 * np.pi**2 * 10.0
        self.omega    = np.sqrt(self.GM / self.r**3)

        # Trail history: one deque per non-BH particle
        self.MAX_TRAIL = 200
        self.trails = [deque(maxlen=self.MAX_TRAIL)
                       for _ in range(self.sim.N - 1)]

    # ── HUD overlay ───────────────────────────────────────────
    _HUD_W       = 280
    _HUD_H       = 226
    # [−] button screen coords (12 px panel margin + PIL inset 18)
    _BTN_MINUS_X0, _BTN_MINUS_X1 = 30,  64   # 12+18, 12+52
    _BTN_PLUS_X0,  _BTN_PLUS_X1  = 240, 274  # 12+228, 12+262
    _BTN_Y0,       _BTN_Y1       = 206, 232  # 12+194, 12+220
    _SPAWN_R_MIN = 0.1
    _SPAWN_R_MAX = 3.0

    def _setup_hud(self):
        """One-time: create shader, VAO and initial texture."""
        W, H = self._HUD_W, self._HUD_H

        # Quad vertices in NDC anchored to top-left with 12 px margin
        mx = 12 / 640.0
        my = 12 / 360.0
        l  = -1.0 + mx
        r  =  l + W / 640.0
        t  =  1.0 - my
        b  =  t - H / 360.0
        verts = np.array([
            l, b, 0, 0,
            r, b, 1, 0,
            l, t, 0, 1,
            r, t, 1, 1,
        ], dtype=np.float32)

        hud_vert = """
        #version 330
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 v_tc;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            v_tc = in_texcoord;
        }
        """
        hud_frag = """
        #version 330
        uniform sampler2D hud_tex;
        in vec2 v_tc;
        out vec4 fragColor;
        void main() { fragColor = texture(hud_tex, v_tc); }
        """
        self.hud_prog = self.ctx.program(
            vertex_shader=hud_vert, fragment_shader=hud_frag
        )
        vbo = self.ctx.buffer(verts)
        self.hud_vao = self.ctx.vertex_array(
            self.hud_prog, [(vbo, '2f 2f', 'in_position', 'in_texcoord')]
        )
        self.hud_prog['hud_tex'].value = 0

        # Allocate texture (filled by _rebuild_hud_tex)
        self.hud_tex = self.ctx.texture((W, H), 4)
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._rebuild_hud_tex()

    def _rebuild_hud_tex(self):
        """Redraw the HUD into the GPU texture — called once at init and on slider change."""
        W, H = self._HUD_W, self._HUD_H
        img  = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Panel background + border
        draw.rectangle([0, 0, W-1, H-1], fill=(8, 10, 22, 190))
        draw.rectangle([0, 0, W-1, H-1], outline=(70, 110, 180, 220), width=1)

        try:
            ft         = "/System/Library/Fonts/Helvetica.ttc"
            font_title = ImageFont.truetype(ft, 13)
            font_body  = ImageFont.truetype(ft, 11)
        except Exception:
            font_title = ImageFont.load_default()
            font_body  = ImageFont.load_default()

        # Title
        draw.text((10, 7), "Controls", font=font_title, fill=(150, 195, 255, 255))
        draw.line([(10, 23), (W-10, 23)], fill=(55, 95, 160, 200), width=1)

        # Key-binding rows
        entries = [
            ("Left drag",    "Rotate camera"),
            ("Scroll",       "Zoom in / out"),
            ("Right-click",  "Spawn star at cursor"),
            ("N",            "Add random star"),
            ("X",            "Remove last star"),
            ("H",            "Toggle this panel"),
        ]
        y = 29
        for key, desc in entries:
            draw.text((10,  y), key,  font=font_body, fill=(255, 205, 55, 255))
            draw.text((118, y), desc, font=font_body, fill=(195, 210, 225, 255))
            y += 23

        # ── Radius buttons section ────────────────────────────────
        sep_y = y + 3
        draw.line([(10, sep_y), (W-10, sep_y)], fill=(55, 95, 160, 200), width=1)

        label_y = sep_y + 7
        draw.text((10, label_y), "Random Star Radius  ([ / ])",
                  font=font_body, fill=(150, 195, 255, 255))

        btn_y0, btn_y1 = label_y + 18, label_y + 44   # PIL coords
        # [−] button
        draw.rectangle([18, btn_y0, 52, btn_y1], fill=(40, 55, 90, 220),
                       outline=(90, 130, 200, 220), width=1)
        draw.text((28, btn_y0 + 5), "−", font=font_title, fill=(220, 230, 255, 255))
        # value label centred
        val_str = f"{self._spawn_r:.2f} AU"
        draw.text((72, btn_y0 + 6), val_str, font=font_body, fill=(255, 205, 55, 255))
        # [+] button
        draw.rectangle([228, btn_y0, 262, btn_y1], fill=(40, 55, 90, 220),
                       outline=(90, 130, 200, 220), width=1)
        draw.text((238, btn_y0 + 5), "+", font=font_title, fill=(220, 230, 255, 255))

        # range hint
        hint_y = btn_y1 + 4
        draw.text((18, hint_y),
                  f"range  {self._SPAWN_R_MIN:.1f} – {self._SPAWN_R_MAX:.1f} AU  |  step 0.1",
                  font=font_body, fill=(100, 115, 140, 200))

        # Flip: PIL top-left → OpenGL bottom-left
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        self.hud_tex.write(img.tobytes())

    # ── World → Screen projection ─────────────────────────────
    def world_to_screen(self, x, y):
        """แปลง AU coordinates → normalized screen (0-1)"""
        scale = 0.12    # zoom: 1 AU = 0.12 ของจอ
        sx    = 0.5 + x * scale
        sy    = 0.5 + y * scale * self.aspect_ratio
        return sx, sy
    # ── Camera matrix ──────────────────────────────────────────────
    def get_view_projection(self):
        # แปลงองศา → radian
        az  = np.radians(self.cam_azimuth)
        el  = np.radians(self.cam_elevation)

        # ตำแหน่ง camera บน sphere รอบ target
        cam_x = self.cam_target[0] + self.cam_distance * np.cos(el) * np.sin(az)
        cam_y = self.cam_target[1] + self.cam_distance * np.sin(el)
        cam_z = self.cam_target[2] + self.cam_distance * np.cos(el) * np.cos(az)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # View matrix (lookAt)
        forward = self.cam_target - cam_pos
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0.0, 1.0, 0.0])
        right    = np.cross(forward, world_up)
        right    = right / np.linalg.norm(right)
        up       = np.cross(right, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] =  right
        view[1, :3] =  up
        view[2, :3] = -forward
        view[0, 3]  = -np.dot(right,   cam_pos)
        view[1, 3]  = -np.dot(up,      cam_pos)
        view[2, 3]  =  np.dot(forward, cam_pos)

        # Projection matrix (perspective)
        fov    = np.radians(45.0)
        aspect = 1280 / 720
        near, far = 0.01, 100.0
        f = 1.0 / np.tan(fov / 2)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0

        return (proj @ view).T   # column-major สำหรับ OpenGL

    # ── Main render loop ──────────────────────────────────────
    def on_render(self, time, frame_time):
        self.sim.integrate(self.sim.t + 0.003)
        self.frame += 1
        t_now = self.frame * 0.003

        scale = 0.5   # AU → world units

        # ── Update star trails ────────────────────────────────────
        n_stars = self.sim.N - 1
        while len(self.trails) < n_stars:
            self.trails.append(deque(maxlen=self.MAX_TRAIL))
        for i, p in enumerate(self.sim.particles[1:]):
            self.trails[i].append((p.x * scale, p.y * scale, p.z * scale))

        # ── คำนวณ camera matrix ──
        mvp = self.get_view_projection()

        star_colors_list = [
            (0.0, 1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        ]

        # ── Pass 1: render scene → FBO ────────────────────────────
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # ── Trail lines ───────────────────────────────────────────
        for i, trail in enumerate(self.trails[:n_stars]):
            if len(trail) < 2:
                continue
            pts  = np.array(trail, dtype=np.float32)
            ages = np.linspace(0.0, 1.0, len(pts), dtype=np.float32).reshape(-1, 1)
            data = np.hstack([pts, ages])
            vbo  = self.ctx.buffer(data.tobytes())
            vao  = self.ctx.vertex_array(
                self.trail_prog, [(vbo, '3f f', 'in_pos', 'in_age')]
            )
            self.trail_prog['color'].value = star_colors_list[i % len(star_colors_list)]
            self.trail_prog['mvp'].write(mvp.tobytes())
            vao.render(moderngl.LINE_STRIP)
            vao.release()
            vbo.release()

        # accretion disk — แปลง 3D → NDC ผ่าน MVP
        theta_now = self.theta + self.omega * t_now
        dx = self.r * np.cos(theta_now)
        dy = self.r * np.sin(theta_now) * np.cos(self.tilt)
        dz = self.r * np.sin(theta_now) * np.sin(self.tilt)

        disk_3d = np.column_stack([
            dx * scale,
            dy * scale,
            dz * scale,
        ]).astype(np.float32)

        # Doppler color
        v_los  = np.sqrt(self.GM / self.r) * np.sin(theta_now)
        v_norm = v_los / (np.max(np.abs(v_los)) + 1e-9)

        for mask, col in [
            (v_norm > 0,  (0.2, 0.6, 1.0, 0.7)),
            (v_norm <= 0, (1.0, 0.3, 0.0, 0.7)),
        ]:
            pts = disk_3d[mask]
            if len(pts) == 0:
                continue
            vbo = self.ctx.buffer(pts.tobytes())
            vao = self.ctx.vertex_array(
                self.scene_prog, [(vbo, '3f', 'in_pos')]   # 3f แทน 2f
            )
            self.scene_prog['color'].value = col
            self.scene_prog['mvp'].write(mvp.tobytes())
            self.scene_prog['point_size'].value = 2.5
            vao.render(moderngl.POINTS)
            vao.release()
            vbo.release()

        # วาดดาว
        for i, p in enumerate(self.sim.particles[1:]):
            sx = np.array(
                [[p.x * scale, p.y * scale, p.z * scale]],
                dtype=np.float32
            )
            vbo = self.ctx.buffer(sx.tobytes())
            vao = self.ctx.vertex_array(
                self.scene_prog, [(vbo, '3f', 'in_pos')]
            )
            self.scene_prog['color'].value = star_colors_list[i % len(star_colors_list)]
            self.scene_prog['mvp'].write(mvp.tobytes())
            self.scene_prog['point_size'].value = 6.0
            vao.render(moderngl.POINTS)
            vao.release()
            vbo.release()

        # ── Pass 2: lensing shader → screen ───────────────────────
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Project BH world position → screen UV using the same MVP as the scene
        bh = self.sim.particles[0]
        bh_world = np.array([bh.x * scale, bh.y * scale, bh.z * scale, 1.0])
        # mvp = (proj @ view).T for OpenGL; mvp.T = proj @ view for Python projection
        bh_clip  = mvp.T @ bh_world
        bh_ndc   = bh_clip[:2] / bh_clip[3]
        bh_sx    = float((bh_ndc[0] + 1.0) / 2.0)
        bh_sy    = float((bh_ndc[1] + 1.0) / 2.0)

        # Scale Schwarzschild radius with distance (0.225 / dist = 0.045 at default dist=5)
        rs_screen = float(0.225 / self.cam_distance)

        self.lensing_prog['bh_screen_pos'].value   = (bh_sx, bh_sy)
        self.lensing_prog['schwarzschild_r'].value  = rs_screen
        self.lensing_prog['resolution'].value       = (1280.0, 720.0)
        self.lensing_prog['cam_az'].value           = float(np.radians(self.cam_azimuth))
        self.lensing_prog['cam_el'].value           = float(np.radians(self.cam_elevation))

        self.star_tex.use(location=0)
        self.scene_tex.use(location=1)
        self.lensing_prog['star_texture'].value  = 0
        self.lensing_prog['scene_texture'].value = 1

        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # ── HUD overlay ───────────────────────────────────────────────
        if self._show_hud:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.hud_tex.use(location=0)
            self.hud_vao.render(moderngl.TRIANGLE_STRIP)

    # ── Spawn helpers ─────────────────────────────────────────────
    def _screen_to_orbital_plane(self, sx, sy):
        """Convert screen pixel → AU position on the y=0 orbital plane. Returns None if no intersection."""
        ndc_x =  (sx / 1280.0) * 2.0 - 1.0
        ndc_y = -(sy / 720.0)  * 2.0 + 1.0   # screen Y is flipped vs NDC

        aspect   = 1280.0 / 720.0
        tan_hfov = np.tan(np.radians(22.5))   # tan(fov/2) for fov=45°

        az = np.radians(self.cam_azimuth)
        el = np.radians(self.cam_elevation)
        ca, sa = np.cos(az), np.sin(az)
        ce, se = np.cos(el), np.sin(el)

        right   = np.array([ca,       0.0, -sa      ])
        up      = np.array([-sa * se, ce,  -ca * se ])
        forward = np.array([-ce * sa, -se, -ce * ca ])

        ray_dir  = right * (ndc_x * tan_hfov * aspect) + up * (ndc_y * tan_hfov) + forward
        ray_dir /= np.linalg.norm(ray_dir)

        cam_pos = np.array([
            self.cam_distance * ce * sa,
            self.cam_distance * se,
            self.cam_distance * ce * ca,
        ])

        # Intersect ray with y=0 plane
        if abs(ray_dir[1]) < 1e-6:
            return None
        t = -cam_pos[1] / ray_dir[1]
        if t < 0.01:
            return None

        world_hit = cam_pos + t * ray_dir
        return world_hit / 0.5   # world units → AU (scale factor = 0.5)

    def _spawn_star(self, sx, sy):
        """Spawn a gravitationally-bound star at the screen-clicked orbital-plane position."""
        pos_au = self._screen_to_orbital_plane(sx, sy)
        if pos_au is None:
            return

        r = float(np.sqrt(pos_au[0]**2 + pos_au[2]**2))
        if r < 0.05:   # too close to the black hole
            return

        GM_bh  = 4.0 * np.pi**2 * 10.0
        v_circ = np.sqrt(GM_bh / r)
        tang   = np.array([-pos_au[2], 0.0, pos_au[0]]) / r   # CCW tangent in y=0 plane

        self.sim.add(
            m  = 1e-6,
            x  = float(pos_au[0]),
            y  = float(pos_au[1]),
            z  = float(pos_au[2]),
            vx = float(tang[0] * v_circ),
            vy = 0.0,
            vz = float(tang[2] * v_circ),
        )
        self._n_spawned += 1
        print(f"Spawned star at r={r:.3f} AU  (total spawned: {self._n_spawned})")

    def _add_random_star(self):
        """Add a star at the slider-set radius with a random angle."""
        r     = float(self._spawn_r)
        theta = float(np.random.uniform(0, 2 * np.pi))
        x_au, z_au = r * np.cos(theta), r * np.sin(theta)

        GM_bh  = 4.0 * np.pi**2 * 10.0
        v_circ = np.sqrt(GM_bh / r)
        tang   = np.array([-z_au, 0.0, x_au]) / r

        self.sim.add(
            m  = 1e-6,
            x  = x_au, y  = 0.0, z  = z_au,
            vx = float(tang[0] * v_circ),
            vy = 0.0,
            vz = float(tang[2] * v_circ),
        )
        self._n_spawned += 1
        print(f"Random star at r={r:.3f} AU  (total spawned: {self._n_spawned})")

    def _remove_last_spawned(self):
        """Remove the most recently spawned user star."""
        if self._n_spawned > 0:
            self.sim.remove(self.sim.N - 1)
            self._n_spawned -= 1
            print(f"Removed last spawned star  (remaining spawned: {self._n_spawned})")

    # ── Radius button helpers ─────────────────────────────────────
    def _in_btn(self, x, y, x0, x1):
        return x0 <= x <= x1 and self._BTN_Y0 <= y <= self._BTN_Y1

    def _change_radius(self, delta):
        self._spawn_r = round(
            np.clip(self._spawn_r + delta, self._SPAWN_R_MIN, self._SPAWN_R_MAX), 2
        )
        self._rebuild_hud_tex()

    # ── Mouse events ──────────────────────────────────────────────
    def on_mouse_press_event(self, x, y, button):
        if button == 4:   # right-click → spawn star
            self._spawn_star(x, y)
        elif self._show_hud and self._in_btn(x, y, self._BTN_MINUS_X0, self._BTN_MINUS_X1):
            self._change_radius(-0.1)
        elif self._show_hud and self._in_btn(x, y, self._BTN_PLUS_X0, self._BTN_PLUS_X1):
            self._change_radius(+0.1)
        else:
            self._mouse_down = True
            self._last_mouse = (x, y)

    def on_mouse_release_event(self, _x, _y, button):
        if button != 4:
            self._mouse_down = False

    def on_mouse_drag_event(self, _x, _y, dx, dy):
        if not self._mouse_down:
            return
        sensitivity       = 0.3
        self.cam_azimuth  -= dx * sensitivity
        self.cam_elevation = np.clip(
            self.cam_elevation + dy * sensitivity, -89.0, 89.0
        )

    def on_mouse_scroll_event(self, _x_offset, y_offset):
        self.cam_distance = np.clip(
            self.cam_distance - y_offset * 0.3, 0.5, 20.0
        )

    def on_key_event(self, key, action, _modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.N:
                self._add_random_star()
            elif key == self.wnd.keys.X:
                self._remove_last_spawned()
            elif key == self.wnd.keys.H:
                self._show_hud = not self._show_hud
            elif key == 91:   # [ — decrease radius
                self._change_radius(-0.1)
            elif key == 93:   # ] — increase radius
                self._change_radius(+0.1)

if __name__ == '__main__':
    mglw.run_window_config(BlackHoleApp)