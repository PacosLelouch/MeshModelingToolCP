#include "objmodel.h"

#include <iostream>
#include <glm.hpp>


// Reference: https://github.com/tinyobjloader/tinyobjloader/blob/master/loader_example.cc
ObjModel::ObjModel()
{
}

ObjModel::~ObjModel()
{
}

bool ObjModel::loadObj(const std::string & filename)
{
	std::cout << "Loading " << filename << std::endl;
	std::string warn;
	std::string err;
	// Do not triangulate.
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str(), nullptr, false);

	if (!warn.empty()) {
		std::cout << "WARN: " << warn << std::endl;
	}

	if (!err.empty()) {
		std::cerr << "ERR: " << err << std::endl;
	}

	if (!ret) {
		std::cerr << "Failed to load obj" << std::endl;
		return false;
	}

	std::cout << "[LoadOBJ] # of shapes in .obj : " << shapes.size() << std::endl;
	std::cout << "[LoadOBJ] # of materials in .obj : " << materials.size() << std::endl;
	
	generateDrawables();

	return true;
}

bool ObjModel::copyObj(const ObjModel& model)
{
	attrib = model.attrib;
	materials = model.materials;
	shapes = model.shapes;
	generateDrawables();
	return false;
}

void ObjModel::drawObj()
{
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_CULL_FACE);
	for (const auto& drawable : mDrawables)
	{
		glBindVertexArray(drawable->VAO);
		glDrawArrays(GL_TRIANGLES, 0, drawable->elementCount);
	}
	glDisable(GL_CULL_FACE);
}

void ObjModel::generateDrawables()
{
	mDrawables.clear();

	// Iterate all shapes
	for (size_t s = 0; s < shapes.size(); ++s)
	{
		std::unique_ptr<Drawable> drawable = std::make_unique<Drawable>();
		std::vector<glm::vec3> vertexBuffer; // pos, color, normal

		generateDrawablePolygons(drawable, vertexBuffer, shapes[s]);

		glBindVertexArray(drawable->VAO);

		// Allocate space and upload the data from CPU to GPU. (pos, nor, col)
		glBindBuffer(GL_ARRAY_BUFFER, drawable->VBO);
		glBufferData(GL_ARRAY_BUFFER, vertexBuffer.size() * sizeof(glm::vec3), vertexBuffer.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
		glEnableVertexAttribArray(2);

		mDrawables.push_back(std::move(drawable));
	}
}

void ObjModel::generateDrawableTriangles(std::unique_ptr<Drawable>& drawable, std::vector<glm::vec3>& vertexBuffer, const tinyobj::shape_t& shape, const std::vector<tinyobj::index_t>* alternateIndices)
{
	bool hasColor = !attrib.colors.empty();
	bool hasNormal = !attrib.normals.empty();

	const std::vector<tinyobj::index_t>& indices = alternateIndices ? *alternateIndices : shape.mesh.indices;

	drawable->elementCount = static_cast<int>(indices.size());
	// Construct buffer
	// Iterate all faces
	vertexBuffer.reserve(drawable->elementCount * 3);
	for (size_t f = 0; f < drawable->elementCount / 3; ++f)
	{
		int idx0 = indices[3 * f + 0].vertex_index;
		int idx1 = indices[3 * f + 1].vertex_index;
		int idx2 = indices[3 * f + 2].vertex_index;
		glm::vec3 pos0 = glm::vec3(attrib.vertices[3 * idx0 + 0], attrib.vertices[3 * idx0 + 1], attrib.vertices[3 * idx0 + 2]);
		glm::vec3 pos1 = glm::vec3(attrib.vertices[3 * idx1 + 0], attrib.vertices[3 * idx1 + 1], attrib.vertices[3 * idx1 + 2]);
		glm::vec3 pos2 = glm::vec3(attrib.vertices[3 * idx2 + 0], attrib.vertices[3 * idx2 + 1], attrib.vertices[3 * idx2 + 2]);

		glm::vec3 nor0, nor1, nor2;
		if (hasNormal)
		{
			idx0 = indices[3 * f + 0].normal_index;
			idx1 = indices[3 * f + 1].normal_index;
			idx2 = indices[3 * f + 2].normal_index;
			nor0 = glm::vec3(attrib.normals[3 * idx0 + 0], attrib.normals[3 * idx0 + 1], attrib.normals[3 * idx0 + 2]);
			nor1 = glm::vec3(attrib.normals[3 * idx1 + 0], attrib.normals[3 * idx1 + 1], attrib.normals[3 * idx1 + 2]);
			nor2 = glm::vec3(attrib.normals[3 * idx2 + 0], attrib.normals[3 * idx2 + 1], attrib.normals[3 * idx2 + 2]);
		}
		else
		{
			// Calculate normal
			glm::vec3 nor = glm::normalize(glm::cross(pos1 - pos0, pos2 - pos0));
			nor0 = nor1 = nor2 = nor;
		}

		glm::vec3 col0, col1, col2;
		if (hasColor)
		{
			idx0 = indices[3 * f + 0].vertex_index;
			idx1 = indices[3 * f + 1].vertex_index;
			idx2 = indices[3 * f + 2].vertex_index;
			col0 = glm::vec3(attrib.colors[3 * idx0 + 0], attrib.colors[3 * idx0 + 1], attrib.colors[3 * idx0 + 2]);
			col1 = glm::vec3(attrib.colors[3 * idx1 + 0], attrib.colors[3 * idx1 + 1], attrib.colors[3 * idx1 + 2]);
			col2 = glm::vec3(attrib.colors[3 * idx2 + 0], attrib.colors[3 * idx2 + 1], attrib.colors[3 * idx2 + 2]);
		}
		else
		{
			glm::vec3 col = glm::vec3(1.0f, 0.0f, 0.0f);
			col0 = col1 = col2 = col;
		}

		//glm::vec3 col = glm::vec3(0, 0, 1);

		vertexBuffer.push_back(pos0);
		vertexBuffer.push_back(nor0);
		vertexBuffer.push_back(col0);

		vertexBuffer.push_back(pos1);
		vertexBuffer.push_back(nor1);
		vertexBuffer.push_back(col1);

		vertexBuffer.push_back(pos2);
		vertexBuffer.push_back(nor2);
		vertexBuffer.push_back(col2);
	}
}

void ObjModel::generateDrawablePolygons(std::unique_ptr<Drawable>& drawable, std::vector<glm::vec3>& vertexBuffer, const tinyobj::shape_t& shape)
{
	std::vector<tinyobj::index_t> indices;
	size_t startIdx = 0;
	for (size_t i = 0; i < shape.mesh.num_face_vertices.size(); ++i)
	{
		int numVert = shape.mesh.num_face_vertices[i];

		if (numVert < 3)
		{
			continue;
		}
		else
		{
			for (int j = 1; j + 1 < numVert; ++j)
			{
				indices.push_back(shape.mesh.indices[startIdx + 0]);
				indices.push_back(shape.mesh.indices[startIdx + j]);
				indices.push_back(shape.mesh.indices[startIdx + j + 1]);
			}
		}
		startIdx += numVert;
	}

	generateDrawableTriangles(drawable, vertexBuffer, shape, &indices);
}
