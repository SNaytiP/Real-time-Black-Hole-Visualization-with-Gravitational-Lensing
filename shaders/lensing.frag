#version 330 core

uniform sampler2D scene_texture;   // ภาพ scene ที่ render แล้ว
uniform sampler2D star_texture;    // starfield background

uniform vec2  bh_screen_pos;       // ตำแหน่งหลุมดำบนจอ (0-1)
uniform float bh_mass;             // มวลหลุมดำ (ควบคุม lensing strength)
uniform float schwarzschild_r;     // Schwarzschild radius ในหน่วย screen
uniform vec2  resolution;          // ขนาดจอ
uniform float cam_az;              // camera azimuth (radians)
uniform float cam_el;              // camera elevation (radians)

in vec2 v_texcoord;
out vec4 fragColor;

const float PI  = 3.14159265358979;
const float TAU = 6.28318530717959;
// tan(fov/2) for fov = 45 degrees
const float TAN_HALF_FOV = 0.41421356;

// ── Gravitational lensing deflection ──────────────────────────
// weak-field approximation: α = 2rs / b
vec2 lensing_deflection(vec2 uv, vec2 bh_pos, float rs) {
    vec2  delta  = uv - bh_pos;
    delta.x     *= resolution.x / resolution.y;
    float b      = length(delta);
    float b_safe = max(b, rs * 1.5);
    float alpha  = min(2.0 * rs / (b_safe * b_safe), 0.8);
    vec2 deflection  = normalize(delta) * alpha;
    deflection.x    /= resolution.x / resolution.y;
    return deflection;
}

// ── Convert screen UV → world-space ray based on camera orientation ──
vec3 uv_to_world_ray(vec2 uv) {
    float aspect   = resolution.x / resolution.y;
    vec2  ndc      = uv * 2.0 - 1.0;

    // View-space ray (camera looks down -Z)
    vec3 ray_view = vec3(ndc.x * TAN_HALF_FOV * aspect,
                         ndc.y * TAN_HALF_FOV,
                         -1.0);

    // Camera basis vectors from azimuth / elevation
    float ca = cos(cam_az), sa = sin(cam_az);
    float ce = cos(cam_el), se = sin(cam_el);

    vec3 right   = vec3( ca,       0.0, -sa);
    vec3 up      = vec3(-sa * se,  ce,  -ca * se);
    vec3 forward = vec3(-ce * sa, -se,  -ce * ca);

    return normalize(right * ray_view.x + up * ray_view.y + forward);
}

// ── World-space ray → equirectangular UV ──────────────────────
vec2 ray_to_uv(vec3 ray) {
    float phi   = atan(ray.x, ray.z);             // [-PI, PI]
    float theta = asin(clamp(ray.y, -1.0, 1.0));  // [-PI/2, PI/2]
    return vec2(phi / TAU + 0.5, theta / PI + 0.5);
}

// ── Photon ring (Einstein ring) ────────────────────────────────
float photon_ring(vec2 uv, vec2 bh_pos, float rs) {
    vec2  d         = uv - bh_pos;
    d.x            *= resolution.x / resolution.y;
    float dist      = length(d);
    float ring_r    = rs * 1.5;
    float ring_width= rs * 0.4;
    float ring      = smoothstep(ring_r - ring_width, ring_r, dist)
                    * smoothstep(ring_r + ring_width, ring_r, dist);
    float glow      = exp(-pow((dist - ring_r) / (rs * 0.8), 2.0)) * 0.6;
    return ring + glow;
}

// ── Event horizon ──────────────────────────────────────────────
float event_horizon(vec2 uv, vec2 bh_pos, float rs) {
    vec2 d = uv - bh_pos;
    d.x   *= resolution.x / resolution.y;
    return smoothstep(rs * 1.1, rs * 0.9, length(d));
}

void main() {
    vec2 uv = v_texcoord;

    // Apply lensing deflection in screen space
    vec2 deflection = lensing_deflection(uv, bh_screen_pos, schwarzschild_r);
    vec2 lensed_uv  = uv + deflection;

    // Convert lensed screen UV → world ray → spherical UV for starfield
    vec3 ray       = uv_to_world_ray(lensed_uv);
    vec2 star_uv   = ray_to_uv(ray);
    vec4 star_color = texture(star_texture, star_uv);

    // Scene (orbiting stars, accretion disk) — no lensing applied
    vec4 scene_color = texture(scene_texture, uv);

    vec4 color = star_color;

    // Photon ring
    float ring      = photon_ring(uv, bh_screen_pos, schwarzschild_r);
    vec3  ring_color = mix(
        vec3(1.0, 0.6, 0.1),
        vec3(0.8, 0.9, 1.0),
        smoothstep(0.0, 1.0, uv.y)
    );
    color.rgb += ring_color * ring * 2.5;

    // Event horizon — solid black
    float eh  = event_horizon(uv, bh_screen_pos, schwarzschild_r);
    color.rgb  = mix(color.rgb, vec3(0.0), eh);

    // Overlay scene particles / stars
    color.rgb  = mix(color.rgb, scene_color.rgb, scene_color.a * 0.9);

    fragColor = color;
}
