#version 330 core
in vec3 nor;
in vec3 lightDir;
in vec3 vertCol;

layout(location = 0) out vec4 fCol;

uniform float uAlpha = 1.0;

void main()
{
    fCol = vec4(0.5 * (nor + 1.0), 1.0);
} 