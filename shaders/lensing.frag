#version 330 core

uniform sampler2D scene_texture;   // ภาพ scene ที่ render แล้ว
uniform sampler2D star_texture;    // starfield background

uniform vec2  bh_screen_pos;       // ตำแหน่งหลุมดำบนจอ (0-1)
uniform float bh_mass;             // มวลหลุมดำ (ควบคุม lensing strength)
uniform float schwarzschild_r;     // Schwarzschild radius ในหน่วย screen
uniform vec2  resolution;          // ขนาดจอ

in vec2 v_texcoord;
out vec4 fragColor;

// ── Gravitational lensing deflection ──────────────────────────
// weak-field approximation: α = 2rs / b
// b = impact parameter (ระยะห่างจาก photon ถึง black hole)
vec2 lensing_deflection(vec2 uv, vec2 bh_pos, float rs) {

    // vector จาก pixel ไปหาหลุมดำ
    vec2  delta     = uv - bh_pos;

    // ปรับ aspect ratio
    delta.x        *= resolution.x / resolution.y;

    float b         = length(delta);   // impact parameter

    // ป้องกัน division by zero ใกล้ event horizon
    float b_safe    = max(b, rs * 1.5);

    // deflection angle: α = 2rs / b²  (screen-space approximation)
    float alpha     = 2.0 * rs / (b_safe * b_safe);

    // clamp ไม่ให้ distort มากเกินไป
    alpha           = min(alpha, 0.8);

    // deflect UV ไปในทิศทางของหลุมดำ (แสงโค้งเข้าหา)
    vec2 deflection = normalize(delta) * alpha;
    deflection.x   /= resolution.x / resolution.y;

    return deflection;
}

// ── Photon ring (Einstein ring) ────────────────────────────────
// วงแสงที่เกิดรอบ photon sphere ที่ r = 1.5 rs
float photon_ring(vec2 uv, vec2 bh_pos, float rs) {
    vec2  d    = uv - bh_pos;
    d.x       *= resolution.x / resolution.y;
    float dist = length(d);

    // photon sphere อยู่ที่ 1.5 * rs
    float ring_r     = rs * 1.5;
    float ring_width = rs * 0.4;
    float ring       = smoothstep(ring_r - ring_width, ring_r, dist)
                     * smoothstep(ring_r + ring_width, ring_r, dist);

    // glow รอบ photon ring
    float glow = exp(-pow((dist - ring_r) / (rs * 0.8), 2.0)) * 0.6;

    return ring + glow;
}

// ── Event horizon (พื้นที่มืดสนิท) ────────────────────────────
float event_horizon(vec2 uv, vec2 bh_pos, float rs) {
    vec2  d = uv - bh_pos;
    d.x    *= resolution.x / resolution.y;
    return smoothstep(rs * 1.1, rs * 0.9, length(d));
}

void main() {
    vec2 uv = v_texcoord;

    // ── คำนวณ lensing deflection ──
    vec2 deflection  = lensing_deflection(uv, bh_screen_pos, schwarzschild_r);

    // UV ของ background หลังถูก deflect
    vec2 lensed_uv   = uv + deflection;

    // ── Sample textures ──
    // background starfield ที่ถูก lens
    vec4 star_color  = texture(star_texture, lensed_uv);

    // scene (ดาวโคจร, accretion disk) ไม่ถูก lens
    vec4 scene_color = texture(scene_texture, uv);

    // ── รวม layers ──
    vec4 color = star_color;

    // เพิ่ม photon ring
    float ring  = photon_ring(uv, bh_screen_pos, schwarzschild_r);
    vec3  ring_color = mix(
        vec3(1.0, 0.6, 0.1),   // สีส้มด้านล่าง (Doppler blueshift ใน disk)
        vec3(0.8, 0.9, 1.0),   // สีขาว-ฟ้าด้านบน
        smoothstep(0.0, 1.0, uv.y)
    );
    color.rgb  += ring_color * ring * 2.5;

    // event horizon = ดำสนิท
    float eh    = event_horizon(uv, bh_screen_pos, schwarzschild_r);
    color.rgb   = mix(color.rgb, vec3(0.0), eh);

    // overlay scene (particles, stars)
    color.rgb   = mix(color.rgb, scene_color.rgb, scene_color.a * 0.9);

    fragColor   = color;
}