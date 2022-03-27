#version 330 core
in vec3 nor;
in vec3 lightDir;
in vec3 vertCol;

layout(location = 0) out vec4 fCol;

uniform float uMaxError = 1.0;

vec3 hsv2rgb(in vec3 inHSV)
{
	float minValue = 0.0;
	float chroma = 0.0;
	float hDash = 0.0;
	float x = 0.0;
	vec3 RGB = vec3(0.0);

	chroma = inHSV.y * inHSV.z;
	hDash = clamp(inHSV.x, 0.0, 360.0) / 60.0;
	float hDashQ = floor(hDash / 2.0) * 2.0;
	float hDashR = hDash - hDashQ;
	x = chroma * (1.0 - abs(hDashR - 1.0));
	//x = chroma * (1.0 - abs(modf(hDash, 2.0) - 1.0));

	if(hDash < 1.0)
	{
		RGB.x = chroma;
		RGB.y = x;
	}
	else if(hDash < 2.0)
	{
		RGB.x = x;
		RGB.y = chroma;
	}
	else if(hDash < 3.0)
	{
		RGB.y = chroma;
		RGB.z = x;
	}
	else if(hDash < 4.0)
	{
		RGB.y = x;
		RGB.z = chroma;
	}
	else if(hDash < 5.0)
	{
		RGB.x = x;
		RGB.z = chroma;
	}
	else if(hDash <= 6.0)
	{
		RGB.x = chroma;
		RGB.z = x;
	}

	minValue = inHSV.z - chroma;

	RGB.x += minValue;
	RGB.y += minValue;
	RGB.z += minValue;

	return RGB;
}

vec3 heatValue2hsv(in vec3 heatValue)
{
	vec3 HSV = vec3(1.0, 1.0, 1.0);
	HSV.x = 240.0 * max(1.0 - abs(heatValue.x) / abs(heatValue.y), 0.0);
	return HSV;
}

void main()
{
	vec3 heatValue = vec3(vertCol.x, uMaxError, 1.0);
	vec3 HSV = heatValue2hsv(heatValue);
	vec3 RGB = hsv2rgb(HSV);
	
	//vec3 RGB = vec3(heatValue);
    fCol = vec4(RGB, 1.0);
} 