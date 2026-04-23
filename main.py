import moderngl
import moderngl_window as mglw
import numpy as np
import rebound
import reboundx
from pathlib import Path
from starfield import generate_starfield
from PIL import Image
import io

class BlackHoleApp(mglw.WindowConfig):
    title          = "Black Hole — Gravitational Lensing"
    window_size    = (1280, 720)
    aspect_ratio   = 1280 / 720
    resource_dir   = Path(__file__).parent

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_simulation()
        self._setup_starfield()
        self._setup_shaders()
        self._setup_framebuffer()
        self._setup_particles()
        self.frame = 0

        # ── Camera state ──────────────────────────────────────────
        self.cam_azimuth   = 45.0    # หมุนรอบแกน Y (องศา)
        self.cam_elevation = 25.0    # เงยขึ้น-ลง (องศา)
        self.cam_distance  = 5.0     # ระยะห่าง
        self.cam_target    = np.array([0.0, 0.0, 0.0])  # จุดที่มอง

        # ── Mouse state ───────────────────────────────────────────
        self._mouse_down   = False
        self._last_mouse   = (0, 0)

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
        in vec2 in_pos;
        uniform mat4 mvp;
        void main() { gl_Position = mvp * vec4(in_pos, 0.0, 1.0);
                       gl_PointSize = 4.0; }
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

        # ── คำนวณ camera matrix ──
        mvp = self.get_view_projection()

        # ── Pass 1: render scene → FBO ────────────────────────────
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # accretion disk — แปลง 3D → NDC ผ่าน MVP
        theta_now = self.theta + self.omega * t_now
        dx = self.r * np.cos(theta_now)
        dy = self.r * np.sin(theta_now) * np.cos(self.tilt)
        dz = self.r * np.sin(theta_now) * np.sin(self.tilt)

        # scale AU → world units
        scale = 0.5
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
            self.ctx.point_size = 2.5
            vao.render(moderngl.POINTS)
            vao.release()
            vbo.release()

        # วาดดาว
        star_colors_list = [
            (0.0, 1.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        ]
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
            self.ctx.point_size = 6.0
            vao.render(moderngl.POINTS)
            vao.release()
            vbo.release()

        # ── Pass 2: lensing shader → screen ───────────────────────
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        bh    = self.sim.particles[0]
        self.lensing_prog['bh_screen_pos'].value   = (0.5, 0.5)  # BH อยู่กลางจอ
        self.lensing_prog['schwarzschild_r'].value  = 0.045
        self.lensing_prog['resolution'].value       = (1280.0, 720.0)

        self.star_tex.use(location=0)
        self.scene_tex.use(location=1)
        self.lensing_prog['star_texture'].value  = 0
        self.lensing_prog['scene_texture'].value = 1

        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    # ── Mouse events ──────────────────────────────────────────────
    def on_mouse_press(self, x, y, button):
        self._mouse_down = True
        self._last_mouse = (x, y)

    def on_mouse_release(self, x, y, button):
        self._mouse_down = False

    def on_mouse_drag(self, x, y, dx, dy):
        if not self._mouse_down:
            return
        sensitivity       = 0.3
        self.cam_azimuth  -= dx * sensitivity
        self.cam_elevation = np.clip(
            self.cam_elevation + dy * sensitivity, -89.0, 89.0
        )

    def on_mouse_scroll(self, x, y, x_offset, y_offset):
        self.cam_distance = np.clip(
            self.cam_distance - y_offset * 0.3, 0.5, 20.0
        )

if __name__ == '__main__':
    mglw.run_window_config(BlackHoleApp)