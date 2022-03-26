#version 330

layout (triangles) in;
layout (line_strip, max_vertices = 4) out;

in vec3 gNor[];
in vec3 gLightDir[];
in vec3 gVertCol[];

out vec3 norP;
out vec3 lightDir;
out vec3 vertCol;

void main()
{
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec4 p2 = gl_in[2].gl_Position;

    vec3 p01 = p1.xyz - p0.xyz;
    vec3 p02 = p2.xyz - p0.xyz;
    norP = normalize(cross(p01, p02));

    gl_Position = p0;
    lightDir = gLightDir[0];
    vertCol = gVertCol[0];
    EmitVertex();

    gl_Position = p1;
    lightDir = gLightDir[1];
    vertCol = gVertCol[1];
    EmitVertex();

    gl_Position = p2;
    lightDir = gLightDir[2];
    vertCol = gVertCol[2];
    EmitVertex();

    gl_Position = p0;
    lightDir = gLightDir[0];
    vertCol = gVertCol[0];
    EmitVertex();
    
    EndPrimitive();
}