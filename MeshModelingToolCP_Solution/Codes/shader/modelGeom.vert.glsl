#version 330 core
layout (location = 0) in vec3 vPos;
layout (location = 1) in vec3 vNor;
layout (location = 2) in vec3 vCol;

uniform mat4 uModel;
uniform mat3 uModelInvTr; // The inverse transpose of the model matrix.
uniform mat4 uProjView;
uniform vec3 uLightPos;

out vec3 gPosW;
out vec3 gNor;
out vec3 gLightDir;
out vec3 gVertCol;

void main()
{
    gPosW = vPos;
    vec4 modelPos = uModel * vec4(vPos, 1.0);
    gLightDir = uLightPos - vec3(modelPos);
    gVertCol = vCol;

    gNor = normalize(uModelInvTr * vNor);

    gl_Position = uProjView * modelPos;
}