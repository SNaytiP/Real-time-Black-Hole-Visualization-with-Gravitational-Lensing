#version 330 core

in vec3 in_pos;          /* 3D position */
uniform mat4 mvp;

void main() {
    gl_Position  = mvp * vec4(in_pos, 1.0);
    gl_PointSize = 4.0;
}