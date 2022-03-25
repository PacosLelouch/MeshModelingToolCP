#pragma once
#include <tiny_obj_loader.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <memory>
#include "drawable.h"
#include "ObjToEigenConverter.h"

// Only support triangle mesh now
class ObjModel
{
public:
	friend class AAShapeUp::ObjToEigenConverter;

	ObjModel();

	~ObjModel();

	// This function should be called after the initialization of OpenGL
	bool loadObj(const std::string& filename);

	bool copyObj(const ObjModel& model);

	void drawObj();

protected:
	void generateDrawables();

protected:
	std::vector<tinyobj::shape_t> shapes; 
	std::vector<tinyobj::material_t> materials;
	tinyobj::attrib_t attrib;

	std::vector<std::unique_ptr<Drawable>> mDrawables;
};