#version 330

layout (triangles) in;
layout (line_strip, max_vertices = 3) out;

uniform float thickness = 0.01;
uniform vec2 screenSize;

void main()
{
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec4 p2 = gl_in[2].gl_Position;

    gl_Position = p0;
    EmitVertex();
    gl_Position = p1;
    EmitVertex();
    gl_Position = p2;
    EmitVertex();
    
    EndPrimitive();
}