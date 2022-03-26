#version 330

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 gPosW[];
in vec3 gNor[];
in vec3 gLightDir[];
in vec3 gVertCol[];

out vec3 nor;
out vec3 lightDir;
out vec3 vertCol;

void main()
{
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec4 p2 = gl_in[2].gl_Position;

    vec3 v0 = gPosW[0];
    vec3 v1 = gPosW[1];
    vec3 v2 = gPosW[2];

    vec3 v01 = v1.xyz - v0.xyz;
    vec3 v02 = v2.xyz - v0.xyz;
    nor = normalize(cross(v01, v02));

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

    EndPrimitive();
}